package main

import (
	"encoding/json"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/rs/zerolog"
)

type PaymentHandler struct {
	payments    sync.Map
	logger      zerolog.Logger
	delayMs     int
	failureRate float64
}

func NewPaymentHandler(logger zerolog.Logger) *PaymentHandler {
	delayMs := 100
	if v := os.Getenv("PAYMENT_DELAY_MS"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed >= 0 {
			delayMs = parsed
		}
	}

	failureRate := 0.05
	if v := os.Getenv("PAYMENT_FAILURE_RATE"); v != "" {
		if parsed, err := strconv.ParseFloat(v, 64); err == nil && parsed >= 0 && parsed <= 1 {
			failureRate = parsed
		}
	}

	return &PaymentHandler{
		logger:      logger,
		delayMs:     delayMs,
		failureRate: failureRate,
	}
}

func (h *PaymentHandler) Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "healthy"})
}

func (h *PaymentHandler) Ready(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (h *PaymentHandler) ProcessPayment(w http.ResponseWriter, r *http.Request) {
	var req PaymentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.logger.Warn().Err(err).Msg("invalid payment request body")
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
		return
	}

	if req.OrderID <= 0 || req.Amount <= 0 || req.UserID <= 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "order_id, amount, and user_id must be positive"})
		return
	}

	start := time.Now()

	time.Sleep(time.Duration(h.delayMs) * time.Millisecond)

	status := "completed"
	if rand.Float64() < h.failureRate {
		status = "failed"
	}

	paymentProcessingDuration.Observe(time.Since(start).Seconds())
	paymentsTotal.WithLabelValues(status).Inc()

	payment := Payment{
		ID:        uuid.New().String(),
		OrderID:   req.OrderID,
		Amount:    req.Amount,
		Status:    status,
		CreatedAt: time.Now().UTC(),
	}

	h.payments.Store(payment.ID, payment)

	h.logger.Info().
		Str("payment_id", payment.ID).
		Int("order_id", payment.OrderID).
		Float64("amount", payment.Amount).
		Str("status", status).
		Msg("payment processed")

	code := http.StatusCreated
	if status == "failed" {
		code = http.StatusPaymentRequired
	}
	writeJSON(w, code, payment)
}

func (h *PaymentHandler) GetPayment(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing payment id"})
		return
	}

	val, ok := h.payments.Load(id)
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "payment not found"})
		return
	}

	writeJSON(w, http.StatusOK, val.(Payment))
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}
