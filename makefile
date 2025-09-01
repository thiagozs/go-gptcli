# Simple Makefile for gptcli

GO ?= go
BIN_DIR := bin
BIN_NAME := gptcli
BIN := $(BIN_DIR)/$(BIN_NAME)
PKG := .             # compila o módulo atual (melhor que apontar só main.go)

LDFLAGS := -s -w
BUILD_FLAGS := -trimpath -ldflags '$(LDFLAGS)'

INSTALL_DIR ?= $(HOME)/.local/bin

.PHONY: all build install uninstall clean fmt tidy test run docker-build docker-run

all: build

$(BIN): go.mod $(shell find . -name '*.go')
	@mkdir -p $(BIN_DIR)
	$(GO) build $(BUILD_FLAGS) -o $(BIN) $(PKG)

build: $(BIN)

install: build
	@mkdir -p $(INSTALL_DIR)
	install -m 0755 $(BIN) $(INSTALL_DIR)/$(BIN_NAME)
	@echo "Installed to $(INSTALL_DIR)/$(BIN_NAME)"

uninstall:
	@rm -f $(INSTALL_DIR)/$(BIN_NAME)
	@echo "Removed $(INSTALL_DIR)/$(BIN_NAME)"

clean:
	rm -rf $(BIN_DIR)

fmt:
	$(GO) fmt ./...

tidy:
	$(GO) mod tidy

test:
	$(GO) test ./...

run: build
	@test -n "$$OPENAI_API_KEY" || (echo "Set OPENAI_API_KEY"; exit 1)
	./$(BIN) --model gpt-5-mini "hello from Makefile"

# Docker helpers
IMAGE ?= gptcli:latest

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	@test -n "$$OPENAI_API_KEY" || (echo "Set OPENAI_API_KEY"; exit 1)
	docker run --rm -it \
		-e OPENAI_API_KEY="$$OPENAI_API_KEY" \
		$(IMAGE) --model gpt-5-mini "teste docker"
