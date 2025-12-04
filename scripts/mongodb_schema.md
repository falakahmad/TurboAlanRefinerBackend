# MongoDB Schema Documentation

This document describes the MongoDB collections and their schemas used in the Alan Refiner application.

## Collections

### 1. users

Stores user account information.

```javascript
{
  _id: ObjectId,
  id: String (UUID), // Unique identifier for the user
  email: String (unique, lowercase),
  password_hash: String,
  first_name: String,
  last_name: String,
  settings: {
    openai_api_key: String,
    openai_model: String,
    target_scanner_risk: Number,
    min_word_ratio: Number
  },
  role: String ('user' | 'admin'),
  is_active: Boolean,
  google_id: String (optional),
  avatar_url: String (optional),
  created_at: Date,
  last_login_at: Date (optional),
  updated_at: Date (optional)
}
```

**Indexes:**
- `email` (unique)
- `role`
- `is_active`

### 2. usage_stats

Tracks OpenAI API usage statistics per user per day.

```javascript
{
  _id: ObjectId,
  user_id: String (UUID, optional),
  request_count: Number,
  tokens_in: Number,
  tokens_out: Number,
  cost: Number,
  model: String,
  job_id: String (optional),
  date: String (ISO date format),
  created_at: Date,
  updated_at: Date
}
```

**Indexes:**
- `user_id` + `date` (unique compound index)
- `user_id`
- `date`
- `job_id`

### 3. schema_usage_stats

Tracks which refinement schemas are used by each user.

```javascript
{
  _id: ObjectId,
  user_id: String (UUID),
  schema_id: String,
  usage_count: Number,
  last_used_at: Date,
  created_at: Date,
  updated_at: Date
}
```

**Indexes:**
- `user_id` + `schema_id` (unique compound index)
- `user_id`
- `schema_id`

### 4. jobs

Stores refinement job information.

```javascript
{
  _id: ObjectId,
  id: String (UUID), // Unique identifier for the job
  user_id: String (UUID, optional),
  file_name: String,
  file_id: String,
  status: String ('pending' | 'processing' | 'completed' | 'failed' | 'cancelled'),
  total_passes: Number,
  current_pass: Number,
  model: String,
  metadata: Object,
  created_at: Date,
  updated_at: Date
}
```

**Indexes:**
- `id` (unique)
- `user_id`
- `created_at` (descending)
- `status`

### 5. job_events

Stores detailed events for each job (pass starts, completions, errors, etc.).

```javascript
{
  _id: ObjectId,
  job_id: String (UUID),
  event_type: String,
  message: String,
  pass_number: Number (optional),
  details: Object,
  created_at: Date
}
```

**Indexes:**
- `job_id`
- `created_at`

### 6. system_logs

Stores application-wide system logs.

```javascript
{
  _id: ObjectId,
  user_id: String (UUID, optional),
  action: String (optional),
  details: String (optional),
  level: String ('INFO' | 'ERROR' | 'WARNING' | 'DEBUG'),
  logger_name: String,
  message: String,
  module: String (optional),
  function_name: String (optional),
  line_number: Number (optional),
  traceback: String (optional),
  metadata: Object,
  ip_address: String (optional),
  user_agent: String (optional),
  created_at: Date
}
```

**Indexes:**
- `user_id`
- `action`
- `created_at` (descending)
- `level`

## Index Creation

All indexes are automatically created by the MongoDB service on initialization. The `_create_indexes()` method in `backend/app/core/mongodb_db.py` handles this.

## Notes

- All dates are stored as MongoDB Date objects (UTC)
- The `id` field (UUID) is used as the application-level identifier, while `_id` is MongoDB's internal ObjectId
- Collections are created automatically when first document is inserted
- Indexes are created automatically on service initialization


