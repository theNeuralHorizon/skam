package main

import "time"

// Order represents an order in the system.
type Order struct {
	ID        int       `json:"id"`
	UserID    int       `json:"user_id"`
	ProductID int       `json:"product_id"`
	Quantity  int       `json:"quantity"`
	Total     float64   `json:"total"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
}

// CreateOrderRequest is the payload for creating a new order.
type CreateOrderRequest struct {
	UserID    int `json:"user_id"`
	ProductID int `json:"product_id"`
	Quantity  int `json:"quantity"`
}

// Validate checks that required fields are present and valid.
func (r CreateOrderRequest) Validate() string {
	if r.UserID <= 0 {
		return "user_id must be a positive integer"
	}
	if r.ProductID <= 0 {
		return "product_id must be a positive integer"
	}
	if r.Quantity <= 0 {
		return "quantity must be a positive integer"
	}
	return ""
}

// PaymentRequest is sent to the payment service.
type PaymentRequest struct {
	OrderID int     `json:"order_id"`
	Amount  float64 `json:"amount"`
	UserID  int     `json:"user_id"`
}

// PaymentResponse is the response from the payment service.
type PaymentResponse struct {
	ID     string `json:"id"`
	Status string `json:"status"`
}

// NotificationRequest is sent to the notification service.
type NotificationRequest struct {
	UserID  int    `json:"user_id"`
	Type    string `json:"type"`
	Message string `json:"message"`
}
