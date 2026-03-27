FROM golang:1.22-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /payment-service .

FROM alpine:3.20

RUN apk --no-cache add ca-certificates
COPY --from=builder /payment-service /payment-service

EXPOSE 8084

ENTRYPOINT ["/payment-service"]
