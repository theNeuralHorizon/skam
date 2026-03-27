package main

import (
	"net/http"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	httpRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "http_requests_total",
		Help: "Total number of HTTP requests.",
	}, []string{"method", "path", "status"})

	httpRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "http_request_duration_seconds",
		Help:    "Duration of HTTP requests in seconds.",
		Buckets: prometheus.DefBuckets,
	}, []string{"method", "path"})

	notificationsSentTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "notifications_sent_total",
		Help: "Total number of notifications sent.",
	}, []string{"type", "status"})

	notificationDeliveryDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "notification_delivery_duration_seconds",
		Help:    "Duration of notification delivery in seconds.",
		Buckets: []float64{0.01, 0.025, 0.05, 0.1, 0.2, 0.5},
	})
)

type statusRecorder struct {
	http.ResponseWriter
	statusCode int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.statusCode = code
	r.ResponseWriter.WriteHeader(code)
}

func prometheusMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(rec, r)

		routePattern := chi.RouteContext(r.Context()).RoutePattern()
		if routePattern == "" {
			routePattern = r.URL.Path
		}

		duration := time.Since(start).Seconds()
		status := strconv.Itoa(rec.statusCode)

		httpRequestsTotal.WithLabelValues(r.Method, routePattern, status).Inc()
		httpRequestDuration.WithLabelValues(r.Method, routePattern).Observe(duration)
	})
}
