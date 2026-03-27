package main

import (
	"context"
	"encoding/json"
	"math/rand"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
	"github.com/rs/zerolog"
)

type NotificationHandler struct {
	rdb    *redis.Client
	logger zerolog.Logger
}

func NewNotificationHandler(rdb *redis.Client, logger zerolog.Logger) *NotificationHandler {
	return &NotificationHandler{
		rdb:    rdb,
		logger: logger,
	}
}

func (h *NotificationHandler) Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "healthy"})
}

func (h *NotificationHandler) Ready(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()

	if err := h.rdb.Ping(ctx).Err(); err != nil {
		h.logger.Error().Err(err).Msg("redis not ready")
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "not ready", "error": "redis unavailable"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (h *NotificationHandler) SendNotification(w http.ResponseWriter, r *http.Request) {
	var req SendNotificationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.logger.Warn().Err(err).Msg("invalid notification request body")
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
		return
	}

	if req.UserID <= 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "user_id must be positive"})
		return
	}
	if req.Type != "email" && req.Type != "sms" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "type must be 'email' or 'sms'"})
		return
	}
	if req.Subject == "" || req.Body == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "subject and body are required"})
		return
	}

	notification := Notification{
		ID:        uuid.New().String(),
		UserID:    req.UserID,
		Type:      req.Type,
		Subject:   req.Subject,
		Status:    "pending",
		CreatedAt: time.Now().UTC(),
	}

	ctx := r.Context()
	data, _ := json.Marshal(notification)
	key := "notification:" + notification.ID

	if err := h.rdb.Set(ctx, key, data, 24*time.Hour).Err(); err != nil {
		h.logger.Error().Err(err).Str("notification_id", notification.ID).Msg("failed to store notification")
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to store notification"})
		return
	}

	start := time.Now()
	delayMs := 50 + rand.Intn(151)
	time.Sleep(time.Duration(delayMs) * time.Millisecond)
	notificationDeliveryDuration.Observe(time.Since(start).Seconds())

	notification.Status = "delivered"
	notificationsSentTotal.WithLabelValues(notification.Type, notification.Status).Inc()

	data, _ = json.Marshal(notification)
	if err := h.rdb.Set(ctx, key, data, 24*time.Hour).Err(); err != nil {
		h.logger.Error().Err(err).Str("notification_id", notification.ID).Msg("failed to update notification status")
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to update notification"})
		return
	}

	h.logger.Info().
		Str("notification_id", notification.ID).
		Int("user_id", notification.UserID).
		Str("type", notification.Type).
		Str("status", notification.Status).
		Msg("notification delivered")

	writeJSON(w, http.StatusCreated, notification)
}

func (h *NotificationHandler) GetNotificationStatus(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing notification id"})
		return
	}

	key := "notification:" + id
	data, err := h.rdb.Get(r.Context(), key).Bytes()
	if err == redis.Nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "notification not found"})
		return
	}
	if err != nil {
		h.logger.Error().Err(err).Str("notification_id", id).Msg("failed to fetch notification")
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to fetch notification"})
		return
	}

	var notification Notification
	if err := json.Unmarshal(data, &notification); err != nil {
		h.logger.Error().Err(err).Msg("failed to unmarshal notification")
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "internal error"})
		return
	}

	writeJSON(w, http.StatusOK, notification)
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}
