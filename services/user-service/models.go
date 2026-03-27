package main

import "time"

// User represents a user account in the system.
type User struct {
	ID           int       `json:"id"`
	Username     string    `json:"username"`
	Email        string    `json:"email"`
	PasswordHash string    `json:"-"`
	CreatedAt    time.Time `json:"created_at"`
}

// CreateUserRequest is the payload for creating a new user.
type CreateUserRequest struct {
	Username string `json:"username"`
	Email    string `json:"email"`
	Password string `json:"password"`
}

// Validate checks that all required fields are present and well-formed.
func (r CreateUserRequest) Validate() map[string]string {
	errors := make(map[string]string)
	if r.Username == "" {
		errors["username"] = "username is required"
	} else if len(r.Username) < 3 || len(r.Username) > 64 {
		errors["username"] = "username must be between 3 and 64 characters"
	}
	if r.Email == "" {
		errors["email"] = "email is required"
	} else if !isValidEmail(r.Email) {
		errors["email"] = "email is not valid"
	}
	if r.Password == "" {
		errors["password"] = "password is required"
	} else if len(r.Password) < 8 {
		errors["password"] = "password must be at least 8 characters"
	}
	if len(errors) == 0 {
		return nil
	}
	return errors
}

// LoginRequest is the payload for authenticating a user.
type LoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

// Validate checks that all required fields are present.
func (r LoginRequest) Validate() map[string]string {
	errors := make(map[string]string)
	if r.Username == "" {
		errors["username"] = "username is required"
	}
	if r.Password == "" {
		errors["password"] = "password is required"
	}
	if len(errors) == 0 {
		return nil
	}
	return errors
}

// LoginResponse is returned on successful authentication.
type LoginResponse struct {
	Token string `json:"token"`
	User  User   `json:"user"`
}

// ErrorResponse is the standard JSON error envelope.
type ErrorResponse struct {
	Error            string            `json:"error"`
	ValidationErrors map[string]string `json:"validation_errors,omitempty"`
}

// isValidEmail performs a basic email format check.
func isValidEmail(email string) bool {
	if len(email) > 254 {
		return false
	}
	at := -1
	for i, c := range email {
		if c == '@' {
			if at != -1 {
				return false // multiple @
			}
			at = i
		}
	}
	if at < 1 || at >= len(email)-1 {
		return false
	}
	domain := email[at+1:]
	dot := false
	for _, c := range domain {
		if c == '.' {
			dot = true
		}
	}
	return dot
}
