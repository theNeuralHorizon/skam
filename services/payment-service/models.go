package main

import "time"

type PaymentRequest struct {
	OrderID int     `json:"order_id"`
	Amount  float64 `json:"amount"`
	UserID  int     `json:"user_id"`
}

type Payment struct {
	ID        string    `json:"id"`
	OrderID   int       `json:"order_id"`
	Amount    float64   `json:"amount"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
}
