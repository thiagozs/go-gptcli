# ---------- builder ----------
FROM golang:1.23-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ENV CGO_ENABLED=0
RUN go build -trimpath -ldflags "-s -w" -o /out/gptcli .


# ---------- runtime ----------
FROM alpine:3.20
RUN apk add --no-cache ca-certificates && adduser -D -h /home/app app
USER app
WORKDIR /home/app
COPY --from=builder /out/gptcli /usr/local/bin/gptcli
ENV OPENAI_API_KEY=""
ENTRYPOINT ["/usr/local/bin/gptcli"]