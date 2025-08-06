# API Reference

Complete API documentation for the test project.

## Authentication

### Login Endpoint

`POST /api/auth/login`

Authenticates a user and returns a JWT token.

### Parameters

- `username` (string, required): User's username
- `password` (string, required): User's password

### Response

```json
{
  "token": "jwt-token-here",
  "expires_in": 3600
}
```

## Data Endpoints

### Get All Items

`GET /api/items`

Returns a list of all items.

### Create Item

`POST /api/items`

Creates a new item in the system.

## Error Handling

All endpoints return standard HTTP status codes and error messages in JSON format.
