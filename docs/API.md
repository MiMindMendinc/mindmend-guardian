# Luna API Reference (Prototype)

Base URL (local default): `http://127.0.0.1:5000`

> Prototype API only. Not intended for unsupervised production child-safety use.

## Authentication

When `LUNA_REQUIRE_AUTH=true`:

```http
Authorization: Bearer <jwt>
```

Obtain a JWT from `/auth_kid` (local development only).

---

## POST `/check_chat`

Scan a chat message for grooming/toxicity signals.

### Request

```json
{
  "message": "sample message text",
  "parent_token": "optional-firebase-token"
}
```

Validation:

- `message` required, string, max 4000 characters
- rejects empty/whitespace-only messages

### Responses

**Safe**

```json
{ "safe": true }
```

**Blocked**

```json
{
  "blocked": true,
  "details": {
    "danger": { "is_flagged": true, "score": 2, "matches": ["alone", "hotel"] },
    "toxicity": { "toxic": false, "polarity": 0.0, "entity_count": 0, "bad_entities": [] }
  }
}
```

**Errors**

- `400` invalid payload
- `401` missing/invalid bearer token (when auth enabled)

---

## POST `/check_location`

Check whether coordinates are outside the configured safe zone.

### Request

```json
{
  "lat": 42.3314,
  "lon": -83.0458,
  "parent_token": "optional-firebase-token"
}
```

### Responses

**Safe**

```json
{ "safe": true }
```

**Alert**

```json
{ "alert": "Outside safe zone" }
```

Alert messages intentionally avoid logging exact coordinates in notification bodies.

---

## POST `/auth_kid`

Issue a short-lived JWT for demo clients.

### Request

```json
{ "user_id": "demo-child-1" }
```

Optional header when `LUNA_AUTH_BOOTSTRAP_TOKEN` is configured:

```http
X-Luna-Bootstrap-Token: <bootstrap-token>
```

### Response

```json
{ "token": "<jwt>" }
```

### Notes

- POST-only (GET is not supported)
- `user_id` is required
- Intended for local development, not anonymous public token vending

---

## Error format

```json
{ "error": "human-readable message" }
```

## Privacy note

Use synthetic or clearly consented test data during development. Do not send real child chat content to prototype endpoints without a completed privacy and safety review.
