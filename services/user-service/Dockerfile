# ---------- Build stage ----------
FROM golang:1.22-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /src

# Cache dependencies.
COPY go.mod go.sum ./
RUN go mod download

# Build the binary.
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /usr/local/bin/user-service .

# ---------- Runtime stage ----------
FROM gcr.io/distroless/static-debian12

COPY --from=builder /usr/local/bin/user-service /user-service

EXPOSE 8081

ENTRYPOINT ["/user-service"]
