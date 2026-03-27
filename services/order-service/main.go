package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	chimw "github.com/go-chi/chi/v5/middleware"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	// Structured JSON logging.
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = zerolog.New(os.Stdout).With().Timestamp().Str("service", "order-service").Logger()

	// Read configuration from environment.
	port := getEnv("PORT", "8083")
	databaseURL := getEnv("DATABASE_URL", "postgres://skam:skam-secret@postgres:5432/orderdb?sslmode=disable")
	redisURL := getEnv("REDIS_URL", "redis:6379")
	paymentServiceURL := getEnv("PAYMENT_SERVICE_URL", "http://payment-service:8084")
	notificationServiceURL := getEnv("NOTIFICATION_SERVICE_URL", "http://notification-service:8085")

	// Connect to PostgreSQL.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	dbPool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to create database pool")
	}
	defer dbPool.Close()

	if err := dbPool.Ping(ctx); err != nil {
		log.Fatal().Err(err).Msg("failed to ping database")
	}
	log.Info().Msg("connected to PostgreSQL")

	// Ensure orders table exists.
	if err := ensureSchema(ctx, dbPool); err != nil {
		log.Fatal().Err(err).Msg("failed to ensure database schema")
	}

	// Connect to Redis.
	rdb := redis.NewClient(&redis.Options{
		Addr: redisURL,
	})
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Fatal().Err(err).Msg("failed to ping redis")
	}
	defer rdb.Close()
	log.Info().Msg("connected to Redis")

	// Create handler.
	handler := NewOrderHandler(dbPool, rdb, paymentServiceURL, notificationServiceURL)

	// Set up Chi router.
	r := chi.NewRouter()
	r.Use(chimw.RequestID)
	r.Use(chimw.RealIP)
	r.Use(chimw.Recoverer)
	r.Use(chimw.Timeout(30 * time.Second))
	r.Use(PrometheusMiddleware)

	// Routes.
	r.Get("/health", handler.HealthCheck)
	r.Get("/ready", handler.ReadinessCheck)
	r.Handle("/metrics", promhttp.Handler())

	r.Post("/orders", handler.CreateOrder)
	r.Get("/orders/{id}", handler.GetOrder)
	r.Get("/orders/user/{user_id}", handler.GetUserOrders)

	// Start server.
	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      r,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Graceful shutdown.
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Info().Str("port", port).Msg("order-service starting")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal().Err(err).Msg("server failed")
		}
	}()

	<-done
	log.Info().Msg("shutting down order-service")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer shutdownCancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Error().Err(err).Msg("server shutdown error")
	}

	log.Info().Msg("order-service stopped")
}

// ensureSchema creates the orders table if it does not exist.
func ensureSchema(ctx context.Context, pool *pgxpool.Pool) error {
	_, err := pool.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS orders (
			id         SERIAL PRIMARY KEY,
			user_id    INTEGER NOT NULL,
			product_id INTEGER NOT NULL,
			quantity   INTEGER NOT NULL,
			total      NUMERIC(12,2) NOT NULL,
			status     VARCHAR(50) NOT NULL DEFAULT 'pending',
			created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
		)
	`)
	if err != nil {
		return err
	}

	// Index for user_id lookups.
	_, err = pool.Exec(ctx, `
		CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id)
	`)
	return err
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
