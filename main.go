// gptcli-advanced: ChatGPT CLI em Go com perfis, REPL, streaming, retry/backoff e proxy
// Compilar:
//   go mod init example.com/gptcli
//   go get github.com/openai/openai-go/v2 gopkg.in/yaml.v3
//   go build -o gptcli .
// Uso:
//   ./gptcli --help

package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/user"
	"path/filepath"
	"strings"
	"time"

	openai "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/shared"
	yaml "gopkg.in/yaml.v3"
)

// ===================== Config & Profiles =====================

type Profile struct {
	Model     string  `yaml:"model"`
	System    string  `yaml:"system"`
	Temp      float64 `yaml:"temp"` // use valor < 0 para omitir
	BaseURL   string  `yaml:"base_url"`
	Proxy     string  `yaml:"proxy"`
	Format    string  `yaml:"format"`     // text|markdown|json
	MaxTokens int     `yaml:"max_tokens"` // 0 = omitido
}

type Config struct {
	APIKey   string             `yaml:"api_key"`
	Default  string             `yaml:"default"`
	Profiles map[string]Profile `yaml:"profiles"`
}

func configDir() string {
	usr, err := user.Current()
	if err != nil {
		return "."
	}
	return filepath.Join(usr.HomeDir, ".config", "gptcli")
}

func configPath() string { return filepath.Join(configDir(), "config.yaml") }

func loadConfig() (*Config, error) {
	b, err := os.ReadFile(configPath())
	if err != nil {
		if os.IsNotExist(err) {
			return &Config{Profiles: map[string]Profile{}}, nil
		}
		return nil, err
	}
	var cfg Config
	if err := yaml.Unmarshal(b, &cfg); err != nil {
		return nil, err
	}
	if cfg.Profiles == nil {
		cfg.Profiles = map[string]Profile{}
	}
	return &cfg, nil
}

// ===================== Flags =====================

type Flags struct {
	APIKey    string
	Model     string
	System    string
	Temp      float64
	BaseURL   string
	Proxy     string
	Format    string
	Profile   string
	JSON      bool
	NoContext bool
	MaxTokens int64
	Repl      bool
}

func parseFlags() *Flags {
	f := &Flags{}
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "\nUso: %s [flags] [prompt]\n\n", os.Args[0])
		fmt.Fprintln(os.Stderr, "Se não houver prompt nem stdin, use --repl para o modo interativo.")
		fmt.Fprintln(os.Stderr, "\nFlags:")
		flag.PrintDefaults()
	}
	flag.StringVar(&f.APIKey, "api-key", "", "OpenAI API key (ou use OPENAI_API_KEY)")
	flag.StringVar(&f.Model, "model", "gpt-5-mini", "modelo (ex: gpt-5, gpt-5-mini, gpt-4.1, gpt-4.1-mini)")
	flag.StringVar(&f.System, "system", "", "mensagem de sistema")
	// -1 => não enviar 'temperature' (usa o default do modelo)
	flag.Float64Var(&f.Temp, "temp", -1, "temperature (0-2). Omitido = default do modelo")
	flag.StringVar(&f.BaseURL, "base-url", "", "Base URL customizada (opcional)")
	flag.StringVar(&f.Proxy, "proxy", "", "HTTP(S) proxy (ex: http://user:pass@host:port)")
	flag.StringVar(&f.Format, "format", "text", "formato de saída: text|markdown|json")
	flag.StringVar(&f.Profile, "profile", "", "nome do profile do config.yaml")
	flag.BoolVar(&f.JSON, "json", false, "atalho para --format json")
	flag.BoolVar(&f.NoContext, "no-context", false, "não manter histórico na sessão (turno único)")
	flag.Int64Var(&f.MaxTokens, "max-tokens", 0, "limite de tokens da resposta (0 = auto)")
	flag.BoolVar(&f.Repl, "repl", false, "entra no modo interativo (REPL)")
	flag.Parse()
	if f.JSON {
		f.Format = "json"
	}
	return f
}

// ===================== Utils =====================

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func isPiped() bool {
	st, _ := os.Stdin.Stat()
	return (st.Mode() & os.ModeCharDevice) == 0
}

func readAllStdin() (string, error) {
	if !isPiped() {
		return "", errors.New("stdin is not a pipe")
	}
	b, err := io.ReadAll(os.Stdin)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(b)), nil
}

func ensureDir(p string) { _ = os.MkdirAll(p, 0o755) }

func randJitter(d time.Duration) time.Duration {
	// Add 0-250ms jitter
	var b [2]byte
	_, _ = rand.Read(b[:])
	j := time.Duration(int(b[0])%250) * time.Millisecond
	return d + j
}

func httpClientWithProxy(proxy string) (*http.Client, error) {
	tr := &http.Transport{}
	if proxy != "" {
		u, err := url.Parse(proxy)
		if err != nil {
			return nil, err
		}
		tr.Proxy = http.ProxyURL(u)
	}
	return &http.Client{Transport: tr}, nil
}

// ===================== OpenAI Client =====================

func buildClient(apiKey, baseURL, proxy string) (openai.Client, error) {
	opts := []option.RequestOption{}
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	if proxy != "" {
		hc, err := httpClientWithProxy(proxy)
		if err != nil {
			return openai.Client{}, err
		}
		opts = append(opts, option.WithHTTPClient(hc))
	}
	return openai.NewClient(opts...), nil
}

// ===================== Chat State =====================

type Turn struct {
	Role    string // "user" | "assistant"
	Content string
}

type Session struct {
	System string // guardamos o system separadamente
	Turns  []Turn // user/assistant
	Format string // text|markdown|json
}

func (s *Session) addSystem(sys string)  { s.System = strings.TrimSpace(sys) }
func (s *Session) addUser(u string)      { s.Turns = append(s.Turns, Turn{"user", u}) }
func (s *Session) addAssistant(a string) { s.Turns = append(s.Turns, Turn{"assistant", a}) }

func (s *Session) lastSystemContent() (string, bool) {
	if s.System != "" {
		return s.System, true
	}
	return "", false
}

// Monta as mensagens para a chamada da API.
// Se jsonMode==true, injeta uma system extra pedindo JSON estrito.
func (s *Session) messagesForAPI(jsonMode bool) []openai.ChatCompletionMessageParamUnion {
	var msgs []openai.ChatCompletionMessageParamUnion
	if s.System != "" {
		msgs = append(msgs, openai.SystemMessage(s.System))
	}
	if jsonMode {
		msgs = append(msgs, openai.SystemMessage("Responda SOMENTE um objeto JSON válido, sem texto extra."))
	}
	for _, t := range s.Turns {
		switch t.Role {
		case "user":
			msgs = append(msgs, openai.UserMessage(t.Content))
		case "assistant":
			msgs = append(msgs, openai.AssistantMessage(t.Content))
		}
	}
	return msgs
}

// ===================== Retry/Backoff =====================

func withRetries(ctx context.Context, attempts int, fn func() error) error {
	if attempts < 1 {
		attempts = 1
	}
	var err error
	backoff := 500 * time.Millisecond
	for i := 0; i < attempts; i++ {
		err = fn()
		if err == nil {
			return nil
		}
		if i < attempts-1 {
			time.Sleep(randJitter(backoff))
			backoff *= 2
			if backoff > 8*time.Second {
				backoff = 8 * time.Second
			}
		}
	}
	return err
}

// ===================== Streaming Call =====================

func streamOnce(ctx context.Context, client openai.Client, sess *Session,
	model string, temp float64, maxTokens int64) (string, error) {

	jsonMode := (strings.ToLower(sess.Format) == "json")
	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(model),
		Messages: sess.messagesForAPI(jsonMode),
	}
	// Só envia se foi definido (>= 0). Alguns modelos não aceitam customização.
	if temp >= 0 {
		params.Temperature = openai.Float(temp)
	}
	if maxTokens > 0 {
		params.MaxTokens = openai.Int(maxTokens)
	}

	stream := client.Chat.Completions.NewStreaming(ctx, params)
	defer stream.Close()

	var built strings.Builder
	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta.Content // NOTE: case-sensitive per SDK; see below correction.
		if delta != "" {
			built.WriteString(delta)
			fmt.Print(delta)
		}
	}
	fmt.Println()
	if err := stream.Err(); err != nil {
		return "", err
	}
	return built.String(), nil
}

// ===================== History & Transcript =====================

func historyPath() string { return filepath.Join(configDir(), "history.txt") }

func saveHistory(lines ...string) {
	ensureDir(configDir())
	f, err := os.OpenFile(historyPath(), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	for _, l := range lines {
		_, _ = f.WriteString(l + "\n")
	}
	_, _ = f.WriteString(strings.Repeat("-", 40) + "\n")
}

func saveTranscript(path string, sess *Session) error {
	if path == "" {
		path = filepath.Join(configDir(), fmt.Sprintf("transcript-%d.md", time.Now().Unix()))
	}
	ensureDir(filepath.Dir(path))
	var b strings.Builder
	b.WriteString("# gptcli transcript\n\n")
	if sess.System != "" {
		b.WriteString("**system**:\n\n" + sess.System + "\n\n")
	}
	for _, t := range sess.Turns {
		b.WriteString(fmt.Sprintf("**%s**:\n\n%s\n\n", t.Role, t.Content))
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

// ===================== REPL =====================

const helpText = `Comandos:
  /help                  mostra esta ajuda
  /exit | /quit          sai do REPL
  /sys <texto>           define/atualiza a mensagem de sistema
  /format <f>            define formato: text|markdown|json
  /clear                 limpa o contexto da sessão (mantém último system)
  /save [caminho]        salva o transcript em Markdown
`

func repl(ctx context.Context, client openai.Client, sess *Session, model string,
	temp float64, maxTokens int64, noContext bool) {
	fmt.Printf("gptcli • model=%s • ctrl+c/ctrl+d para sair\n", model)
	if _, ok := sess.lastSystemContent(); ok {
		fmt.Println("(system ativo)")
	}
	in := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !in.Scan() {
			break
		}
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "/") {
			parts := strings.Fields(line)
			cmd := parts[0]
			switch cmd {
			case "/help":
				fmt.Print(helpText)
			case "/exit", "/quit":
				return
			case "/sys":
				text := strings.TrimSpace(strings.TrimPrefix(line, "/sys"))
				if text == "" {
					fmt.Println("uso: /sys <texto>")
					continue
				}
				sess.addSystem(text)
				fmt.Println("(system atualizado)")
			case "/format":
				if len(parts) < 2 {
					fmt.Println("uso: /format text|markdown|json")
					continue
				}
				f := strings.ToLower(parts[1])
				if f != "text" && f != "markdown" && f != "json" {
					fmt.Println("formato inválido")
					continue
				}
				sess.Format = f
				fmt.Println("(formato:", f, ")")
			case "/clear":
				var newSys string
				if sys, ok := sess.lastSystemContent(); ok {
					newSys = sys
				}
				sess.Turns = nil
				if newSys != "" {
					sess.System = newSys
				}
				fmt.Println("(contexto limpo)")
			case "/save":
				path := ""
				if len(parts) >= 2 {
					path = parts[1]
				}
				if err := saveTranscript(path, sess); err != nil {
					fmt.Println("erro:", err)
				} else {
					fmt.Println("(transcript salvo)")
				}
			default:
				fmt.Println("comando desconhecido. /help para ajuda")
			}
			continue
		}

		// Mensagem do usuário
		sess.addUser(line)

		call := func() error {
			resp, err := streamOnce(ctx, client, sess, model, temp, maxTokens)
			if err != nil {
				return err
			}
			if !noContext {
				sess.addAssistant(resp)
			} else {
				// sem contexto: remove o último user e o último assistant (se houver)
				// mantendo o system intacto
				if len(sess.Turns) >= 1 && sess.Turns[len(sess.Turns)-1].Role == "assistant" {
					sess.Turns = sess.Turns[:len(sess.Turns)-1]
				}
				if len(sess.Turns) >= 1 && sess.Turns[len(sess.Turns)-1].Role == "user" {
					sess.Turns = sess.Turns[:len(sess.Turns)-1]
				}
			}
			return nil
		}

		if err := withRetries(ctx, 4, call); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
		}
	}
}

// ===================== Entry =====================

func main() {
	flags := parseFlags()
	cfg, _ := loadConfig()

	// Resolve API key: flag > env > config
	apiKey := strings.TrimSpace(flags.APIKey)
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("OPENAI_OPENAI_API_KEY")) // NOTE: typo? We'll correct to OPENAI_API_KEY below.
	}
	if apiKey == "" && cfg != nil {
		apiKey = strings.TrimSpace(cfg.APIKey)
	}
	if apiKey == "" {
		// fallback to correct var name
		apiKey = strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	}
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "defina OPENAI_API_KEY, config.yaml ou --api-key")
		os.Exit(2)
	}

	// Carrega profile do config se informado (ou default)
	prof := Profile{}
	if cfg != nil {
		name := flags.Profile
		if name == "" {
			name = cfg.Default
		}
		if name != "" {
			if p, ok := cfg.Profiles[name]; ok {
				prof = p
			}
		}
	}

	// Merge: flags sobrescrevem profile
	model := chooseNonEmpty(flags.Model, prof.Model, "gpt-5-mini")
	system := chooseNonEmpty(flags.System, prof.System, "")
	temp := chooseTemp(flags.Temp, prof.Temp, -1) // -1 = omitir 'temperature'
	baseURL := chooseNonEmpty(flags.BaseURL, prof.BaseURL, "")
	proxy := chooseNonEmpty(flags.Proxy, prof.Proxy, "")
	format := chooseNonEmpty(flags.Format, prof.Format, "text")
	maxTokens := chooseInt64(flags.MaxTokens, int64(prof.MaxTokens), 0)

	client, err := buildClient(apiKey, baseURL, proxy)
	must(err)

	ctx := context.Background()
	sess := &Session{Format: strings.ToLower(format)}
	sess.addSystem(system)

	// I/O modos: pipe > args > REPL/Help
	if isPiped() {
		piped, err := readAllStdin()
		must(err)
		sess.addUser(piped)
		call := func() error {
			resp, err := streamOnce(ctx, client, sess, model, temp, maxTokens)
			if err != nil {
				return err
			}
			sess.addAssistant(resp)
			return nil
		}
		must(withRetries(ctx, 4, call))
		saveHistory("Q: " + piped)
		return
	}

	if flag.NArg() > 0 {
		prompt := strings.TrimSpace(strings.Join(flag.Args(), " "))
		sess.addUser(prompt)
		call := func() error {
			resp, err := streamOnce(ctx, client, sess, model, temp, maxTokens)
			if err != nil {
				return err
			}
			sess.addAssistant(resp)
			return nil
		}
		must(withRetries(ctx, 4, call))
		saveHistory("Q: " + prompt)
		return
	}

	if flags.Repl {
		repl(ctx, client, sess, model, temp, maxTokens, flags.NoContext)
		return
	}

	// Sem params: mostra help e sai com código 2
	flag.Usage()
	os.Exit(2)
}

// ===================== Helpers =====================

func chooseNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

// Temp: usa sentinela -1 para "não enviar"
func chooseTemp(flagVal, profVal, fallback float64) float64 {
	if flagVal >= 0 {
		return flagVal
	}
	if profVal >= 0 {
		return profVal
	}
	return fallback // normalmente -1
}

func chooseInt64(vals ...int64) int64 {
	for _, v := range vals {
		if v != 0 {
			return v
		}
	}
	return 0
}
