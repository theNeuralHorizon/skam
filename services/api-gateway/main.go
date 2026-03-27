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
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

type serviceConfig struct {
	Name string
	URL  string
}

func getEnv(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return fallback
}

func main() {
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = zerolog.New(os.Stdout).With().
		Timestamp().
		Str("service", "api-gateway").
		Logger()

	services := map[string]serviceConfig{
		"/api/users":         {Name: "user-service", URL: getEnv("USER_SERVICE_URL", "http://user-service:8081")},
		"/api/products":      {Name: "product-service", URL: getEnv("PRODUCT_SERVICE_URL", "http://product-service:8082")},
		"/api/orders":        {Name: "order-service", URL: getEnv("ORDER_SERVICE_URL", "http://order-service:8083")},
		"/api/payments":      {Name: "payment-service", URL: getEnv("PAYMENT_SERVICE_URL", "http://payment-service:8084")},
		"/api/notifications": {Name: "notification-service", URL: getEnv("NOTIFICATION_SERVICE_URL", "http://notification-service:8085")},
	}

	r := chi.NewRouter()

	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(zerologMiddleware)
	r.Use(middleware.Recoverer)
	r.Use(PrometheusMiddleware)

	r.Handle("/metrics", promhttp.Handler())
	mountRoutes(r, services)

	port := getEnv("PORT", "8080")
	addr := ":" + port

	srv := &http.Server{
		Addr:         addr,
		Handler:      r,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		log.Info().Str("addr", addr).Msg("starting api gateway")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal().Err(err).Msg("server failed")
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	log.Info().Str("signal", sig.String()).Msg("shutting down")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal().Err(err).Msg("forced shutdown")
	}
	log.Info().Msg("server stopped")
}

func zerologMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)

		next.ServeHTTP(ww, r)

		log.Info().
			Str("method", r.Method).
			Str("path", r.URL.Path).
			Int("status", ww.Status()).
			Int("bytes", ww.BytesWritten()).
			Dur("duration", time.Since(start)).
			Str("remote_addr", r.RemoteAddr).
			Str("request_id", middleware.GetReqID(r.Context())).
			Msg("request completed")
	})
}
