# go-gptcli

CLI em Go para interagir com a API do OpenAI (ou endpoints compatíveis). Fornece formas simples de enviar prompts, trabalhar em modo REPL, usar profiles e salvar transcrições.

## O que tem aqui

- Binário: `bin/gptcli` (quando compilado)
- Arquivo de configuração: `~/.config/gptcli/config.yaml`
- Histórico: `~/.config/gptcli/history.txt`

## Requisitos

- Go 1.18+ instalado
- Uma chave de API válida (ex.: `OPENAI_API_KEY`)

## Como compilar

No diretório do projeto:

```bash
go build -o bin/gptcli
```

Ou usando o Makefile (se existir):

```bash
make build
```

## Uso rápido — exemplos

1. Prompt direto (args):

```bash
./bin/gptcli "Explique polimorfismo em 3 frases"
```

1. Input via pipe (stdin):

```bash
echo "Resuma este texto:" | ./bin/gptcli
```

1. REPL (modo interativo):

```bash
./bin/gptcli --repl
# No REPL, use /help para ver comandos (ex: /sys, /format, /save, /exit)
```

1. Forçar saída JSON (atalho):

```bash
./bin/gptcli --json "Gere um objeto JSON com campos name e age"
```

1. Especificar modelo, temperatura e formato:

```bash
./bin/gptcli --model gpt-4.1 --temp 0.2 --format markdown "Escreva uma breve análise"
```

1. Usar proxy ou Base URL customizada (ex.: servidor compatível com OpenAI):

```bash
./bin/gptcli --proxy "http://user:pass@proxy:3128" --base-url "https://meu-endpoint.local/v1" "Teste"
```

1. Salvar transcript no REPL:

No REPL, rode:

```
/save caminho/opcional.md
```

1. Desabilitar contexto no REPL (turno único):

```bash
./bin/gptcli --repl --no-context
```

## Flags principais

- `--api-key` — fornece a chave da API (fallback: `OPENAI_API_KEY` ou `config.yaml`).
- `--model` — modelo a usar (ex: `gpt-5-mini`, `gpt-4.1`).
- `--system` — mensagem de sistema a ser incluída.
- `--temp` — temperature (0-2). Valor negativo omite o campo e usa o default do modelo.
- `--format` — `text|markdown|json`.
- `--json` — atalho para `--format json`.
- `--proxy` — HTTP(S) proxy.
- `--base-url` — Base URL customizada.
- `--max-tokens` — limite de tokens para a resposta.
- `--repl` — entra no modo interativo.
- `--no-context` — no REPL, não mantém histórico entre prompts.

Para ajuda rápida:

```bash
./bin/gptcli --help
```

## Arquivo de configuração (opcional)

Local: `~/.config/gptcli/config.yaml`

Exemplo de `config.yaml`:

```yaml
api_key: "sk-..."
default: "dev"
profiles:
    dev:
        model: "gpt-5-mini"
        system: "Você é um assistente útil."
        temp: 0.0
        base_url: ""
        proxy: ""
        format: "text"
        max_tokens: 0

    writer:
        model: "gpt-4.1"
        system: "Escreva no estilo de um artigo técnico."
        temp: 0.3
        format: "markdown"
```

Flags na linha de comando sobrescrevem valores do profile.

## Histórico e transcript

- Cada execução grava uma linha em `~/.config/gptcli/history.txt`.
- No REPL, `/save` salva uma transcrição em Markdown (por padrão em `~/.config/gptcli/`).

## Licença

MIT — veja `LICENSE`.

## Contribuições

- Pull requests e issues são bem-vindos. Para mudanças maiores, abra uma issue primeiro explicando a mudança.

## Instalar o exemplo de configuração

Para copiar o exemplo para a sua pasta de configuração (~/.config/gptcli) rode:

```bash
./scripts/install-config.sh
# ou manualmente:
cp examples/config.yaml ~/.config/gptcli/config.yaml && chmod 600 ~/.config/gptcli/config.yaml
```
