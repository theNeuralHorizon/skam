package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
	"github.com/rs/zerolog/log"
)

const (
	redisCacheTTL      = 5 * time.Minute
	httpClientTimeout  = 5 * time.Second
	defaultProductCost = 29.99
)

// OrderHandler holds dependencies for order HTTP handlers.
type OrderHandler struct {
	DB                     *pgxpool.Pool
	Redis                  *redis.Client
	HTTPClient             *http.Client
	PaymentServiceURL      string
	NotificationServiceURL string
}

// NewOrderHandler creates a new OrderHandler with sensible defaults.
func NewOrderHandler(db *pgxpool.Pool, rdb *redis.Client, paymentURL, notificationURL string) *OrderHandler {
	return &OrderHandler{
		DB:    db,
		Redis: rdb,
		HTTPClient: &http.Client{
			Timeout: httpClientTimeout,
		},
		PaymentServiceURL:      paymentURL,
		NotificationServiceURL: notificationURL,
	}
}

// HealthCheck returns service health status.
func (h *OrderHandler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "healthy",
		"service": "order-service",
	})
}

// ReadinessCheck verifies DB and Redis connectivity.
func (h *OrderHandler) ReadinessCheck(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	defer cancel()

	if err := h.DB.Ping(ctx); err != nil {
		log.Error().Err(err).Msg("database readiness check failed")
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"status": "unhealthy",
			"error":  "database unreachable",
		})
		return
	}

	if err := h.Redis.Ping(ctx).Err(); err != nil {
		log.Error().Err(err).Msg("redis readiness check failed")
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"status": "unhealthy",
			"error":  "redis unreachable",
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

// CreateOrder orchestrates the full order creation flow.
func (h *OrderHandler) CreateOrder(w http.ResponseWriter, r *http.Request) {
	var req CreateOrderRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
		return
	}

	if msg := req.Validate(); msg != "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": msg})
		return
	}

	total := float64(req.Quantity) * defaultProductCost

	// Step 1: Insert order with status "pending".
	order, err := h.insertOrder(r.Context(), req, total)
	if err != nil {
		log.Error().Err(err).Msg("failed to insert order")
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to create order"})
		return
	}

	// Step 2: Call payment service.
	paymentResp, err := h.processPayment(r.Context(), order)
	if err != nil {
		log.Error().Err(err).Int("order_id", order.ID).Msg("payment processing failed")
		_ = h.updateOrderStatus(r.Context(), order.ID, "payment_failed")
		order.Status = "payment_failed"
		writeJSON(w, http.StatusPaymentRequired, map[string]string{
			"error":    "payment processing failed",
			"order_id": strconv.Itoa(order.ID),
			"status":   "payment_failed",
		})
		return
	}

	// Step 3: Update order status to "confirmed".
	if paymentResp.Status == "approved" || paymentResp.Status == "success" {
		if err := h.updateOrderStatus(r.Context(), order.ID, "confirmed"); err != nil {
			log.Error().Err(err).Int("order_id", order.ID).Msg("failed to update order status")
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to confirm order"})
			return
		}
		order.Status = "confirmed"
	} else {
		_ = h.updateOrderStatus(r.Context(), order.ID, "payment_failed")
		order.Status = "payment_failed"
		writeJSON(w, http.StatusPaymentRequired, map[string]string{
			"error":    "payment declined",
			"order_id": strconv.Itoa(order.ID),
			"status":   "payment_failed",
		})
		return
	}

	// Step 4: Send notification (best-effort, do not fail the request).
	if err := h.sendNotification(r.Context(), order); err != nil {
		log.Warn().Err(err).Int("order_id", order.ID).Msg("notification failed (best-effort)")
	}

	// Step 5: Cache order in Redis.
	h.cacheOrder(r.Context(), order)

	writeJSON(w, http.StatusCreated, order)
}

// GetOrder retrieves a single order by ID, checking Redis cache first.
func (h *OrderHandler) GetOrder(w http.ResponseWriter, r *http.Request) {
	idStr := chi.URLParam(r, "id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid order id"})
		return
	}

	// Try cache first.
	cacheKey := fmt.Sprintf("order:%d", id)
	cached, err := h.Redis.Get(r.Context(), cacheKey).Result()
	if err == nil {
		redisCacheHitsTotal.Inc()
		redisOperationsTotal.WithLabelValues("GET", "success").Inc()
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Cache", "HIT")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(cached))
		return
	}
	redisCacheMissesTotal.Inc()
	if err != redis.Nil {
		redisOperationsTotal.WithLabelValues("GET", "error").Inc()
		log.Warn().Err(err).Msg("redis GET error")
	}

	// Fallback to database.
	order, err := h.getOrderFromDB(r.Context(), id)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "order not found"})
		return
	}

	// Populate cache for next time.
	h.cacheOrder(r.Context(), order)

	w.Header().Set("X-Cache", "MISS")
	writeJSON(w, http.StatusOK, order)
}

// GetUserOrders lists all orders for a given user.
func (h *OrderHandler) GetUserOrders(w http.ResponseWriter, r *http.Request) {
	userIDStr := chi.URLParam(r, "user_id")
	userID, err := strconv.Atoi(userIDStr)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid user_id"})
		return
	}

	start := time.Now()
	rows, err := h.DB.Query(r.Context(),
		"SELECT id, user_id, product_id, quantity, total, status, created_at FROM orders WHERE user_id = $1 ORDER BY created_at DESC",
		userID,
	)
	dbQueryDuration.WithLabelValues("select_user_orders").Observe(time.Since(start).Seconds())

	if err != nil {
		log.Error().Err(err).Int("user_id", userID).Msg("failed to query user orders")
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to fetch orders"})
		return
	}
	defer rows.Close()

	var orders []Order
	for rows.Next() {
		var o Order
		if err := rows.Scan(&o.ID, &o.UserID, &o.ProductID, &o.Quantity, &o.Total, &o.Status, &o.CreatedAt); err != nil {
			log.Error().Err(err).Msg("failed to scan order row")
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to read orders"})
			return
		}
		orders = append(orders, o)
	}

	if orders == nil {
		orders = []Order{}
	}

	writeJSON(w, http.StatusOK, orders)
}

// --- internal helpers ---

func (h *OrderHandler) insertOrder(ctx context.Context, req CreateOrderRequest, total float64) (*Order, error) {
	start := time.Now()
	var order Order
	err := h.DB.QueryRow(ctx,
		`INSERT INTO orders (user_id, product_id, quantity, total, status, created_at)
		 VALUES ($1, $2, $3, $4, 'pending', NOW())
		 RETURNING id, user_id, product_id, quantity, total, status, created_at`,
		req.UserID, req.ProductID, req.Quantity, total,
	).Scan(&order.ID, &order.UserID, &order.ProductID, &order.Quantity, &order.Total, &order.Status, &order.CreatedAt)
	dbQueryDuration.WithLabelValues("insert_order").Observe(time.Since(start).Seconds())
	if err != nil {
		return nil, err
	}
	return &order, nil
}

func (h *OrderHandler) updateOrderStatus(ctx context.Context, orderID int, status string) error {
	start := time.Now()
	_, err := h.DB.Exec(ctx, "UPDATE orders SET status = $1 WHERE id = $2", status, orderID)
	dbQueryDuration.WithLabelValues("update_order_status").Observe(time.Since(start).Seconds())
	return err
}

func (h *OrderHandler) getOrderFromDB(ctx context.Context, id int) (*Order, error) {
	start := time.Now()
	var order Order
	err := h.DB.QueryRow(ctx,
		"SELECT id, user_id, product_id, quantity, total, status, created_at FROM orders WHERE id = $1", id,
	).Scan(&order.ID, &order.UserID, &order.ProductID, &order.Quantity, &order.Total, &order.Status, &order.CreatedAt)
	dbQueryDuration.WithLabelValues("select_order").Observe(time.Since(start).Seconds())
	if err != nil {
		return nil, err
	}
	return &order, nil
}

func (h *OrderHandler) processPayment(ctx context.Context, order *Order) (*PaymentResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, httpClientTimeout)
	defer cancel()

	payload, err := json.Marshal(PaymentRequest{
		OrderID: order.ID,
		Amount:  order.Total,
		UserID:  order.UserID,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal payment request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		h.PaymentServiceURL+"/payments/process", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("create payment request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call payment service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("payment service returned status %d", resp.StatusCode)
	}

	var paymentResp PaymentResponse
	if err := json.NewDecoder(resp.Body).Decode(&paymentResp); err != nil {
		return nil, fmt.Errorf("decode payment response: %w", err)
	}

	return &paymentResp, nil
}

func (h *OrderHandler) sendNotification(ctx context.Context, order *Order) error {
	ctx, cancel := context.WithTimeout(ctx, httpClientTimeout)
	defer cancel()

	payload, err := json.Marshal(NotificationRequest{
		UserID:  order.UserID,
		Type:    "order_confirmation",
		Message: fmt.Sprintf("Your order #%d has been confirmed. Total: $%.2f", order.ID, order.Total),
	})
	if err != nil {
		return fmt.Errorf("marshal notification: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		h.NotificationServiceURL+"/notifications/send", bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("create notification request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("call notification service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		return fmt.Errorf("notification service returned status %d", resp.StatusCode)
	}

	return nil
}

func (h *OrderHandler) cacheOrder(ctx context.Context, order *Order) {
	data, err := json.Marshal(order)
	if err != nil {
		log.Warn().Err(err).Msg("failed to marshal order for cache")
		return
	}

	cacheKey := fmt.Sprintf("order:%d", order.ID)
	if err := h.Redis.Set(ctx, cacheKey, data, redisCacheTTL).Err(); err != nil {
		redisOperationsTotal.WithLabelValues("SET", "error").Inc()
		log.Warn().Err(err).Msg("failed to cache order in redis")
		return
	}
	redisOperationsTotal.WithLabelValues("SET", "success").Inc()
}

// writeJSON is a helper to send JSON responses.
func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Error().Err(err).Msg("failed to encode JSON response")
	}
}
