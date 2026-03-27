package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rs/zerolog/log"
)

type ProductHandler struct {
	DB *pgxpool.Pool
}

func NewProductHandler(db *pgxpool.Pool) *ProductHandler {
	return &ProductHandler{DB: db}
}

func (h *ProductHandler) Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "healthy",
		"service": "product-service",
	})
}

func (h *ProductHandler) Ready(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	defer cancel()

	if err := h.DB.Ping(ctx); err != nil {
		log.Error().Err(err).Msg("readiness check failed")
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"status": "not ready",
			"error":  "database unavailable",
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (h *ProductHandler) ListProducts(w http.ResponseWriter, r *http.Request) {
	category := r.URL.Query().Get("category")
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))

	if limit <= 0 || limit > 100 {
		limit = 20
	}
	if offset < 0 {
		offset = 0
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	var rows pgx.Rows
	var err error
	start := time.Now()

	if category != "" {
		rows, err = h.DB.Query(ctx,
			`SELECT id, name, price, stock, category, created_at
			 FROM products WHERE category = $1
			 ORDER BY id LIMIT $2 OFFSET $3`,
			category, limit, offset)
		dbQueryDuration.WithLabelValues("list_products_by_category").Observe(time.Since(start).Seconds())
	} else {
		rows, err = h.DB.Query(ctx,
			`SELECT id, name, price, stock, category, created_at
			 FROM products ORDER BY id LIMIT $1 OFFSET $2`,
			limit, offset)
		dbQueryDuration.WithLabelValues("list_products").Observe(time.Since(start).Seconds())
	}
	if err != nil {
		log.Error().Err(err).Msg("failed to query products")
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to fetch products"})
		return
	}
	defer rows.Close()

	products := make([]Product, 0)
	for rows.Next() {
		var p Product
		if err := rows.Scan(&p.ID, &p.Name, &p.Price, &p.Stock, &p.Category, &p.CreatedAt); err != nil {
			log.Error().Err(err).Msg("failed to scan product")
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to read products"})
			return
		}
		products = append(products, p)
	}

	writeJSON(w, http.StatusOK, products)
}

func (h *ProductHandler) GetProduct(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.Atoi(chi.URLParam(r, "id"))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid product id"})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	var p Product
	start := time.Now()
	err = h.DB.QueryRow(ctx,
		`SELECT id, name, price, stock, category, created_at
		 FROM products WHERE id = $1`, id).
		Scan(&p.ID, &p.Name, &p.Price, &p.Stock, &p.Category, &p.CreatedAt)
	dbQueryDuration.WithLabelValues("get_product").Observe(time.Since(start).Seconds())

	if err != nil {
		if err == pgx.ErrNoRows {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "product not found"})
			return
		}
		log.Error().Err(err).Int("id", id).Msg("failed to get product")
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to fetch product"})
		return
	}

	writeJSON(w, http.StatusOK, p)
}

func (h *ProductHandler) CreateProduct(w http.ResponseWriter, r *http.Request) {
	var req CreateProductRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
		return
	}

	if errs := validateCreateProduct(req); len(errs) > 0 {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "validation failed",
			"details": errs,
		})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	var p Product
	start := time.Now()
	err := h.DB.QueryRow(ctx,
		`INSERT INTO products (name, price, stock, category)
		 VALUES ($1, $2, $3, $4)
		 RETURNING id, name, price, stock, category, created_at`,
		req.Name, req.Price, req.Stock, req.Category).
		Scan(&p.ID, &p.Name, &p.Price, &p.Stock, &p.Category, &p.CreatedAt)
	dbQueryDuration.WithLabelValues("create_product").Observe(time.Since(start).Seconds())

	if err != nil {
		log.Error().Err(err).Msg("failed to create product")
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to create product"})
		return
	}

	log.Info().Int("id", p.ID).Str("name", p.Name).Msg("product created")
	writeJSON(w, http.StatusCreated, p)
}

func (h *ProductHandler) UpdateStock(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.Atoi(chi.URLParam(r, "id"))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid product id"})
		return
	}

	var req UpdateStockRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	var p Product
	start := time.Now()
	err = h.DB.QueryRow(ctx,
		`UPDATE products SET stock = $1 WHERE id = $2
		 RETURNING id, name, price, stock, category, created_at`,
		req.Quantity, id).
		Scan(&p.ID, &p.Name, &p.Price, &p.Stock, &p.Category, &p.CreatedAt)
	dbQueryDuration.WithLabelValues("update_stock").Observe(time.Since(start).Seconds())

	if err != nil {
		if err == pgx.ErrNoRows {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "product not found"})
			return
		}
		log.Error().Err(err).Int("id", id).Msg("failed to update stock")
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to update stock"})
		return
	}

	log.Info().Int("id", p.ID).Int("stock", p.Stock).Msg("stock updated")
	writeJSON(w, http.StatusOK, p)
}

func validateCreateProduct(req CreateProductRequest) []string {
	var errs []string
	if strings.TrimSpace(req.Name) == "" {
		errs = append(errs, "name is required")
	}
	if req.Price <= 0 {
		errs = append(errs, "price must be positive")
	}
	if req.Stock < 0 {
		errs = append(errs, "stock cannot be negative")
	}
	if strings.TrimSpace(req.Category) == "" {
		errs = append(errs, "category is required")
	}
	return errs
}

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}
