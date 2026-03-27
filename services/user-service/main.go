package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/rs/zerolog"
)

func main() {
	// ---- Logger ----
	logger := zerolog.New(os.Stdout).With().
		Timestamp().
		Str("service", "user-service").
		Logger()

	// ---- Config ----
	port := getEnv("PORT", "8081")
	databaseURL := getEnv("DATABASE_URL", "postgres://skam:skam-secret@postgres:5432/userdb?sslmode=disable")

	// ---- Database ----
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	poolConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		logger.Fatal().Err(err).Msg("failed to parse database URL")
	}
	poolConfig.MaxConns = 20
	poolConfig.MinConns = 2
	poolConfig.HealthCheckPeriod = 30 * time.Second

	pool, err := pgxpool.NewWithConfig(ctx, poolConfig)
	if err != nil {
		logger.Fatal().Err(err).Msg("failed to create connection pool")
	}
	defer pool.Close()

	if err := pool.Ping(ctx); err != nil {
		logger.Fatal().Err(err).Msg("failed to ping database")
	}
	logger.Info().Msg("connected to PostgreSQL")

	// Ensure the users table exists.
	if err := ensureSchema(ctx, pool); err != nil {
		logger.Fatal().Err(err).Msg("failed to ensure database schema")
	}

	// ---- Update active-connections gauge periodically ----
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			stat := pool.Stat()
			dbActiveConnections.Set(float64(stat.AcquiredConns()))
		}
	}()

	// ---- Handler ----
	handler := NewUserHandler(pool, logger)

	// ---- Router ----
	r := chi.NewRouter()

	// Middleware stack.
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(PrometheusMiddleware)

	// Routes.
	r.Get("/health", handler.Health)
	r.Get("/ready", handler.Ready)
	r.Handle("/metrics", promhttp.Handler())

	r.Post("/users", handler.CreateUser)
	r.Get("/users/{id}", handler.GetUser)
	r.Get("/users", handler.ListUsers)

	r.Post("/auth/login", handler.Login)

	// ---- HTTP Server ----
	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      r,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in a goroutine.
	go func() {
		logger.Info().Str("port", port).Msg("starting HTTP server")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal().Err(err).Msg("HTTP server error")
		}
	}()

	// ---- Graceful Shutdown ----
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	logger.Info().Str("signal", sig.String()).Msg("shutting down")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer shutdownCancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Error().Err(err).Msg("HTTP server shutdown error")
	}

	pool.Close()
	logger.Info().Msg("user-service stopped")
}

// ensureSchema creates the users table if it does not already exist.
func ensureSchema(ctx context.Context, pool *pgxpool.Pool) error {
	_, err := pool.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS users (
			id            SERIAL PRIMARY KEY,
			username      VARCHAR(64)  NOT NULL UNIQUE,
			email         VARCHAR(254) NOT NULL UNIQUE,
			password_hash TEXT         NOT NULL,
			created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
		);
		CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
		CREATE INDEX IF NOT EXISTS idx_users_email    ON users (email);
	`)
	return err
}

// getEnv returns the value of the environment variable or a default.
func getEnv(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}
