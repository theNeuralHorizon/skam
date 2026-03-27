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
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
	"github.com/rs/zerolog"
)

func main() {
	logger := zerolog.New(os.Stdout).With().Timestamp().Str("service", "notification-service").Logger()

	port := os.Getenv("PORT")
	if port == "" {
		port = "8085"
	}

	redisURL := os.Getenv("REDIS_URL")
	if redisURL == "" {
		redisURL = "redis:6379"
	}

	rdb := redis.NewClient(&redis.Options{
		Addr:         redisURL,
		ReadTimeout:  3 * time.Second,
		WriteTimeout: 3 * time.Second,
		DialTimeout:  5 * time.Second,
		PoolSize:     10,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		logger.Warn().Err(err).Msg("redis not available at startup, will retry on requests")
	}

	handler := NewNotificationHandler(rdb, logger)

	r := chi.NewRouter()
	r.Use(middleware.Recoverer)
	r.Use(middleware.RealIP)
	r.Use(prometheusMiddleware)

	r.Get("/health", handler.Health)
	r.Get("/ready", handler.Ready)
	r.Get("/metrics", promhttp.Handler().ServeHTTP)

	r.Post("/notifications/send", handler.SendNotification)
	r.Get("/notifications/status/{id}", handler.GetNotificationStatus)

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      r,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	go func() {
		logger.Info().Str("port", port).Msg("starting notification service")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal().Err(err).Msg("server failed")
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info().Msg("shutting down server")
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Fatal().Err(err).Msg("server forced to shutdown")
	}

	if err := rdb.Close(); err != nil {
		logger.Error().Err(err).Msg("failed to close redis connection")
	}

	logger.Info().Msg("server stopped")
}
