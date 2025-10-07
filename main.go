package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/base64"
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
	APIKey       string
	Model        string
	System       string
	Temp         float64
	BaseURL      string
	Proxy        string
	Format       string
	Profile      string
	JSON         bool
	NoContext    bool
	MaxTokens    int64
	Repl         bool
	Image        bool
	ImageModel   string
	ImageSize    string
	ImageQuality string
	ImageFormat  string
	ImageOut     string
	ImageCount   int
}

func parseFlags() *Flags {
	f := &Flags{}
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "\nUso: %s [flags] [prompt]\n\n", os.Args[0])
		fmt.Fprintln(os.Stderr, "Se não houver prompt nem stdin, use --repl para o modo interativo.")
		fmt.Fprintln(os.Stderr, "Use --image para gerar imagens a partir do prompt.")
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
	flag.BoolVar(&f.Image, "image", false, "gera imagem em vez de texto")
	flag.StringVar(&f.ImageModel, "image-model", "gpt-image-1", "modelo de imagem (ex: gpt-image-1, dall-e-3)")
	flag.StringVar(&f.ImageSize, "image-size", "", "tamanho da imagem (ex: 1024x1024)")
	flag.StringVar(&f.ImageQuality, "image-quality", "", "qualidade da imagem (auto|high|medium|low|hd etc)")
	flag.StringVar(&f.ImageFormat, "image-format", "", "formato para gpt-image-1 (png|jpeg|webp)")
	flag.StringVar(&f.ImageOut, "image-out", "", "arquivo ou diretório destino (default: ./gpt-image-<timestamp>.png)")
	flag.IntVar(&f.ImageCount, "image-count", 1, "quantidade de imagens (1-10)")
	flag.Parse()
	if f.JSON {
		f.Format = "json"
	}
	if f.ImageCount < 1 {
		f.ImageCount = 1
	}
	if f.ImageCount > 10 {
		f.ImageCount = 10
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

// ===================== Image Generation =====================

func promptForImagePrompt() (string, error) {
	if isPiped() {
		text, err := readAllStdin()
		if err != nil {
			return "", err
		}
		if strings.TrimSpace(text) == "" {
			return "", errors.New("stdin vazio; informe um prompt para gerar a imagem")
		}
		return text, nil
	}
	if flag.NArg() > 0 {
		prompt := strings.TrimSpace(strings.Join(flag.Args(), " "))
		if prompt != "" {
			return prompt, nil
		}
	}
	return "", errors.New("forneça um prompt via stdin ou argumento para gerar a imagem")
}

func generateImages(ctx context.Context, client openai.Client, prompt string, flags *Flags, proxy string) error {
	params := openai.ImageGenerateParams{
		Prompt: prompt,
	}
	if model := strings.TrimSpace(flags.ImageModel); model != "" {
		params.Model = openai.ImageModel(model)
	}
	if size := strings.TrimSpace(flags.ImageSize); size != "" {
		params.Size = openai.ImageGenerateParamsSize(size)
	}
	if quality := strings.TrimSpace(flags.ImageQuality); quality != "" {
		params.Quality = openai.ImageGenerateParamsQuality(quality)
	}
	if format := strings.TrimSpace(flags.ImageFormat); format != "" {
		params.OutputFormat = openai.ImageGenerateParamsOutputFormat(format)
	}
	if flags.ImageCount > 1 {
		params.N = openai.Int(int64(flags.ImageCount))
	}

	resp, err := client.Images.Generate(ctx, params)
	if err != nil {
		return err
	}
	if resp == nil || len(resp.Data) == 0 {
		return errors.New("nenhuma imagem retornada pela API")
	}

	defaultFormat := strings.TrimSpace(flags.ImageFormat)
	if defaultFormat == "" && resp.OutputFormat != "" {
		defaultFormat = string(resp.OutputFormat)
	}
	if defaultFormat == "" {
		defaultFormat = "png"
	}

	outPaths, err := prepareImageOutputPaths(strings.TrimSpace(flags.ImageOut), defaultFormat, len(resp.Data))
	if err != nil {
		return err
	}

	var downloadClient *http.Client
	for i, img := range resp.Data {
		target := outPaths[i]

		if err := ensureFileDirectory(target); err != nil {
			return err
		}

		imgExt := defaultFormat
		if ext := detectExtensionFromURL(img.URL); ext != "" {
			imgExt = ext
		}
		currentExt := strings.TrimPrefix(strings.ToLower(filepath.Ext(target)), ".")
		if currentExt == "" && imgExt != "" {
			target = fmt.Sprintf("%s.%s", target, imgExt)
		} else if imgExt != "" && currentExt != imgExt {
			target = strings.TrimSuffix(target, filepath.Ext(target))
			target = fmt.Sprintf("%s.%s", target, imgExt)
		}

		if err := saveGeneratedImage(ctx, img, target, proxy, &downloadClient); err != nil {
			return fmt.Errorf("falha ao salvar imagem %d: %w", i+1, err)
		}
		fmt.Println("Imagem salva em", target)
	}
	return nil
}

func prepareImageOutputPaths(out, format string, count int) ([]string, error) {
	if count < 1 {
		return nil, errors.New("quantidade de imagens inválida")
	}
	format = strings.TrimPrefix(strings.ToLower(format), ".")
	if format == "" {
		format = "png"
	}
	out = strings.TrimSpace(out)
	if out == "" {
		return defaultImagePaths(format, count), nil
	}

	if strings.HasSuffix(out, string(os.PathSeparator)) {
		dir := strings.TrimSuffix(out, string(os.PathSeparator))
		return imagePathsInsideDir(dir, format, count)
	}

	if info, err := os.Stat(out); err == nil {
		if info.IsDir() {
			return imagePathsInsideDir(out, format, count)
		}
	} else if !os.IsNotExist(err) {
		return nil, err
	}

	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(out)), ".")
	prefix := out
	if ext != "" {
		prefix = strings.TrimSuffix(out, filepath.Ext(out))
		format = ext
	}

	if count == 1 {
		if ext == "" {
			return []string{fmt.Sprintf("%s.%s", out, format)}, nil
		}
		return []string{out}, nil
	}

	paths := make([]string, count)
	for i := 0; i < count; i++ {
		paths[i] = fmt.Sprintf("%s-%d.%s", prefix, i+1, format)
	}
	return paths, nil
}

func defaultImagePaths(format string, count int) []string {
	prefix := defaultImageBasename()
	paths := make([]string, count)
	for i := 0; i < count; i++ {
		name := prefix
		if count > 1 {
			name = fmt.Sprintf("%s-%d", prefix, i+1)
		}
		paths[i] = fmt.Sprintf("%s.%s", name, format)
	}
	return paths
}

func imagePathsInsideDir(dir, format string, count int) ([]string, error) {
	if dir == "" {
		dir = "."
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	prefix := defaultImageBasename()
	paths := make([]string, count)
	for i := 0; i < count; i++ {
		name := prefix
		if count > 1 {
			name = fmt.Sprintf("%s-%d", prefix, i+1)
		}
		paths[i] = filepath.Join(dir, fmt.Sprintf("%s.%s", name, format))
	}
	return paths, nil
}

func defaultImageBasename() string {
	return fmt.Sprintf("gpt-image-%s", time.Now().Format("20060102-150405"))
}

func detectExtensionFromURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(u.Path)), ".")
	switch ext {
	case "jpg":
		return "jpeg"
	case "jpeg", "png", "webp":
		return ext
	default:
		return ""
	}
}

func ensureFileDirectory(path string) error {
	dir := filepath.Dir(path)
	if dir == "" || dir == "." || dir == string(os.PathSeparator) {
		return nil
	}
	return os.MkdirAll(dir, 0o755)
}

func saveGeneratedImage(ctx context.Context, img openai.Image, path, proxy string, cache **http.Client) error {
	if img.B64JSON != "" {
		data, err := base64.StdEncoding.DecodeString(img.B64JSON)
		if err != nil {
			return fmt.Errorf("falha ao decodificar imagem base64: %w", err)
		}
		return os.WriteFile(path, data, 0o644)
	}
	if img.URL != "" {
		client := *cache
		if client == nil {
			hc, err := httpClientWithProxy(proxy)
			if err != nil {
				return err
			}
			client = hc
			*cache = hc
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, img.URL, nil)
		if err != nil {
			return err
		}
		resp, err := client.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 300 {
			snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
			return fmt.Errorf("download da imagem falhou (%s): %s", resp.Status, strings.TrimSpace(string(snippet)))
		}
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		return os.WriteFile(path, data, 0o644)
	}
	return errors.New("imagem sem dados (nem base64 nem URL)")
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

	// Aviso amigável: se existir config.yaml mas não houver api_key, lembre o usuário
	if _, err := os.Stat(configPath()); err == nil {
		if cfg != nil && strings.TrimSpace(cfg.APIKey) == "" {
			fmt.Fprintln(os.Stderr, "nota: config.yaml encontrado mas sem 'api_key'. Use OPENAI_API_KEY ou --api-key para fornecer a chave.")
		}
	}

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

	if flags.Image {
		if flags.Repl {
			fmt.Fprintln(os.Stderr, "--image não é compatível com --repl")
			os.Exit(2)
		}
		prompt, err := promptForImagePrompt()
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(2)
		}
		call := func() error {
			return generateImages(ctx, client, prompt, flags, proxy)
		}
		must(withRetries(ctx, 4, call))
		saveHistory("IMG: " + prompt)
		return
	}

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
