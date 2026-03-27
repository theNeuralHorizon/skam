package main

import "time"

type SendNotificationRequest struct {
	UserID  int    `json:"user_id"`
	Type    string `json:"type"`
	Subject string `json:"subject"`
	Body    string `json:"body"`
}

type Notification struct {
	ID        string    `json:"id"`
	UserID    int       `json:"user_id"`
	Type      string    `json:"type"`
	Subject   string    `json:"subject"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
}
