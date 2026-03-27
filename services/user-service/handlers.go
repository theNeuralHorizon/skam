package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rs/zerolog"
	"golang.org/x/crypto/bcrypt"
)

const (
	bcryptCost       = 12
	defaultLimit     = 20
	maxLimit         = 100
	queryTimeout     = 5 * time.Second
)

// UserHandler holds dependencies for HTTP handlers.
type UserHandler struct {
	DB     *pgxpool.Pool
	Logger zerolog.Logger
}

// NewUserHandler creates a UserHandler with the given pool and logger.
func NewUserHandler(db *pgxpool.Pool, logger zerolog.Logger) *UserHandler {
	return &UserHandler{DB: db, Logger: logger}
}

// ---------- Health / Readiness ----------

// Health returns a static health status.
func (h *UserHandler) Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "healthy",
		"service": "user-service",
	})
}

// Ready pings the database to verify connectivity.
func (h *UserHandler) Ready(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()

	if err := h.DB.Ping(ctx); err != nil {
		h.Logger.Error().Err(err).Msg("readiness check failed")
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "database not reachable"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

// ---------- User CRUD ----------

// CreateUser handles POST /users.
func (h *UserHandler) CreateUser(w http.ResponseWriter, r *http.Request) {
	var req CreateUserRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON body"})
		return
	}

	if errs := req.Validate(); errs != nil {
		writeJSON(w, http.StatusUnprocessableEntity, ErrorResponse{
			Error:            "validation failed",
			ValidationErrors: errs,
		})
		return
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcryptCost)
	if err != nil {
		h.Logger.Error().Err(err).Msg("failed to hash password")
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "internal server error"})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), queryTimeout)
	defer cancel()

	var user User
	start := time.Now()
	err = h.DB.QueryRow(ctx,
		`INSERT INTO users (username, email, password_hash)
		 VALUES ($1, $2, $3)
		 RETURNING id, username, email, password_hash, created_at`,
		req.Username, req.Email, string(hash),
	).Scan(&user.ID, &user.Username, &user.Email, &user.PasswordHash, &user.CreatedAt)
	dbQueryDuration.WithLabelValues("insert_user").Observe(time.Since(start).Seconds())

	if err != nil {
		if isDuplicateKeyError(err) {
			writeJSON(w, http.StatusConflict, ErrorResponse{Error: "username or email already exists"})
			return
		}
		h.Logger.Error().Err(err).Msg("failed to insert user")
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "internal server error"})
		return
	}

	h.Logger.Info().Int("user_id", user.ID).Str("username", user.Username).Msg("user created")
	writeJSON(w, http.StatusCreated, user)
}

// GetUser handles GET /users/{id}.
func (h *UserHandler) GetUser(w http.ResponseWriter, r *http.Request) {
	idStr := chi.URLParam(r, "id")
	id, err := strconv.Atoi(idStr)
	if err != nil || id < 1 {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid user id"})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), queryTimeout)
	defer cancel()

	var user User
	start := time.Now()
	err = h.DB.QueryRow(ctx,
		`SELECT id, username, email, password_hash, created_at FROM users WHERE id = $1`, id,
	).Scan(&user.ID, &user.Username, &user.Email, &user.PasswordHash, &user.CreatedAt)
	dbQueryDuration.WithLabelValues("get_user").Observe(time.Since(start).Seconds())

	if err != nil {
		if err == pgx.ErrNoRows {
			writeJSON(w, http.StatusNotFound, ErrorResponse{Error: "user not found"})
			return
		}
		h.Logger.Error().Err(err).Int("user_id", id).Msg("failed to query user")
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "internal server error"})
		return
	}

	writeJSON(w, http.StatusOK, user)
}

// ListUsers handles GET /users with optional ?limit= and ?offset= query params.
func (h *UserHandler) ListUsers(w http.ResponseWriter, r *http.Request) {
	limit, offset := parsePagination(r)

	ctx, cancel := context.WithTimeout(r.Context(), queryTimeout)
	defer cancel()

	start := time.Now()
	rows, err := h.DB.Query(ctx,
		`SELECT id, username, email, password_hash, created_at
		 FROM users ORDER BY id ASC LIMIT $1 OFFSET $2`, limit, offset,
	)
	dbQueryDuration.WithLabelValues("list_users").Observe(time.Since(start).Seconds())

	if err != nil {
		h.Logger.Error().Err(err).Msg("failed to list users")
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "internal server error"})
		return
	}
	defer rows.Close()

	users := make([]User, 0)
	for rows.Next() {
		var u User
		if err := rows.Scan(&u.ID, &u.Username, &u.Email, &u.PasswordHash, &u.CreatedAt); err != nil {
			h.Logger.Error().Err(err).Msg("failed to scan user row")
			writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "internal server error"})
			return
		}
		users = append(users, u)
	}
	if err := rows.Err(); err != nil {
		h.Logger.Error().Err(err).Msg("row iteration error")
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "internal server error"})
		return
	}

	writeJSON(w, http.StatusOK, users)
}

// ---------- Auth ----------

// Login handles POST /auth/login.
func (h *UserHandler) Login(w http.ResponseWriter, r *http.Request) {
	var req LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON body"})
		return
	}

	if errs := req.Validate(); errs != nil {
		writeJSON(w, http.StatusUnprocessableEntity, ErrorResponse{
			Error:            "validation failed",
			ValidationErrors: errs,
		})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), queryTimeout)
	defer cancel()

	var user User
	start := time.Now()
	err := h.DB.QueryRow(ctx,
		`SELECT id, username, email, password_hash, created_at FROM users WHERE username = $1`, req.Username,
	).Scan(&user.ID, &user.Username, &user.Email, &user.PasswordHash, &user.CreatedAt)
	dbQueryDuration.WithLabelValues("login_lookup").Observe(time.Since(start).Seconds())

	if err != nil {
		if err == pgx.ErrNoRows {
			writeJSON(w, http.StatusUnauthorized, ErrorResponse{Error: "invalid credentials"})
			return
		}
		h.Logger.Error().Err(err).Msg("login query failed")
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "internal server error"})
		return
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(req.Password)); err != nil {
		writeJSON(w, http.StatusUnauthorized, ErrorResponse{Error: "invalid credentials"})
		return
	}

	// Generate a demo token (base64-encoded JSON payload).
	tokenPayload := fmt.Sprintf(`{"user_id":%d,"username":"%s","exp":%d}`,
		user.ID, user.Username, time.Now().Add(24*time.Hour).Unix())
	token := base64.URLEncoding.EncodeToString([]byte(tokenPayload))

	h.Logger.Info().Int("user_id", user.ID).Msg("user logged in")
	writeJSON(w, http.StatusOK, LoginResponse{Token: token, User: user})
}

// ---------- Helpers ----------

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func parsePagination(r *http.Request) (limit, offset int) {
	limit = defaultLimit
	offset = 0

	if v := r.URL.Query().Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			limit = n
		}
	}
	if limit > maxLimit {
		limit = maxLimit
	}
	if v := r.URL.Query().Get("offset"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			offset = n
		}
	}
	return limit, offset
}

// isDuplicateKeyError checks for PostgreSQL unique_violation (SQLSTATE 23505).
func isDuplicateKeyError(err error) bool {
	if err == nil {
		return false
	}
	// pgx wraps PG errors; check the error string for the SQLSTATE code.
	return containsString(err.Error(), "23505")
}

func containsString(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
