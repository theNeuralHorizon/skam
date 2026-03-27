package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/rs/zerolog/log"
)

func mountRoutes(r *chi.Mux, services map[string]serviceConfig) {
	r.Get("/health", healthHandler)
	r.Get("/ready", readinessHandler(services))

	for prefix, svc := range services {
		proxy := newReverseProxy(svc.URL, prefix)
		r.Handle(prefix+"/*", proxy)
	}
}

func healthHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	fmt.Fprint(w, `{"status":"healthy"}`)
}

func readinessHandler(services map[string]serviceConfig) http.HandlerFunc {
	client := &http.Client{Timeout: 2 * time.Second}

	return func(w http.ResponseWriter, _ *http.Request) {
		type result struct {
			Name    string `json:"name"`
			Status  string `json:"status"`
			healthy bool
		}

		var (
			mu      sync.Mutex
			wg      sync.WaitGroup
			results []result
		)

		for _, svc := range services {
			wg.Add(1)
			go func(s serviceConfig) {
				defer wg.Done()
				r := result{Name: s.Name, Status: "healthy", healthy: true}

				resp, err := client.Get(s.URL + "/health")
				if err != nil {
					r.Status = "unreachable"
					r.healthy = false
				} else {
					resp.Body.Close()
					if resp.StatusCode >= 400 {
						r.Status = "unhealthy"
						r.healthy = false
					}
				}

				mu.Lock()
				results = append(results, r)
				mu.Unlock()
			}(svc)
		}
		wg.Wait()

		allHealthy := true
		for _, r := range results {
			if !r.healthy {
				allHealthy = false
				break
			}
		}

		w.Header().Set("Content-Type", "application/json")
		status := http.StatusOK
		if !allHealthy {
			status = http.StatusServiceUnavailable
		}
		w.WriteHeader(status)

		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":   map[bool]string{true: "ready", false: "not_ready"}[allHealthy],
			"services": results,
		})
	}
}

func newReverseProxy(targetURL, stripPrefix string) http.Handler {
	target, err := url.Parse(targetURL)
	if err != nil {
		log.Fatal().Err(err).Str("target", targetURL).Msg("invalid proxy target URL")
	}

	proxy := httputil.NewSingleHostReverseProxy(target)
	originalDirector := proxy.Director

	proxy.Director = func(req *http.Request) {
		originalDirector(req)
		req.URL.Path = strings.TrimPrefix(req.URL.Path, stripPrefix)
		if req.URL.Path == "" {
			req.URL.Path = "/"
		}
		req.Host = target.Host
	}

	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Error().
			Err(err).
			Str("target", targetURL).
			Str("path", r.URL.Path).
			Msg("proxy error")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		json.NewEncoder(w).Encode(map[string]string{
			"error":   "service unavailable",
			"message": err.Error(),
		})
	}

	return proxy
}
