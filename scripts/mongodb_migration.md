# MongoDB Migration Guide

This guide helps you migrate from Supabase to MongoDB.

## Prerequisites

1. MongoDB instance (MongoDB Atlas recommended for production)
2. MongoDB connection string
3. Python `pymongo` package installed
4. Node.js `mongodb` package installed (for frontend)

## Setup Steps

### 1. Get MongoDB Connection String

If using MongoDB Atlas:
1. Go to your MongoDB Atlas dashboard
2. Click "Connect" on your cluster
3. Choose "Connect your application"
4. Copy the connection string
5. Replace `<password>` with your database user password

Example:
```
mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
```

### 2. Configure Environment Variables

#### Backend (.env)

Add to `backend/.env`:
```bash
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=alan_refiner
```

#### Frontend (.env.local)

Add to `Frontend/.env.local`:
```bash
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=alan_refiner
```

**Important:** The MongoDB URL should NOT be exposed to the client. It's only used in Next.js API routes (server-side).

### 3. Install Dependencies

#### Backend
```bash
cd backend
pip install pymongo==4.10.1 bcrypt==4.2.0
```

#### Frontend
```bash
cd Frontend
npm install mongodb@^6.3.0
```

### 4. Remove Supabase Dependencies (Optional)

#### Backend
```bash
pip uninstall supabase
```

#### Frontend
```bash
npm uninstall @supabase/supabase-js
```

### 5. Verify Connection

The MongoDB service will automatically:
- Connect to MongoDB on application startup
- Create indexes if they don't exist
- Handle connection pooling
- Retry on connection failures

Check your application logs for:
```
[MongoDB] MongoDB client initialized successfully.
```

## Schema Migration

The MongoDB schema is automatically created when you first insert data. No manual schema creation is needed.

However, if you want to migrate existing data from Supabase:

### Manual Data Migration

1. Export data from Supabase (using Supabase dashboard or SQL)
2. Transform data to match MongoDB schema (see `mongodb_schema.md`)
3. Import into MongoDB using `mongoimport` or MongoDB Compass

Example:
```bash
mongoimport --uri="mongodb+srv://..." --db=alan_refiner --collection=users --file=users.json
```

## Testing

1. Start the backend:
```bash
cd backend
uvicorn app.main:app --reload
```

2. Start the frontend:
```bash
cd Frontend
npm run dev
```

3. Test user registration/login
4. Test refinement job creation
5. Check MongoDB to verify data is being stored

## Troubleshooting

### Connection Issues

- Verify MongoDB URL is correct
- Check network firewall rules (MongoDB Atlas requires IP whitelist)
- Verify database user has read/write permissions

### Index Creation Errors

- Indexes are created automatically on startup
- If errors occur, check MongoDB user permissions
- Indexes can be created manually using MongoDB Compass or `mongo` shell

### Performance

- MongoDB connection pooling is configured automatically
- Adjust `maxPoolSize` and `minPoolSize` in `mongodb_db.py` if needed
- Monitor MongoDB Atlas metrics for connection usage

## Production Considerations

1. **Security:**
   - Never expose MongoDB URL to client-side code
   - Use environment variables for all sensitive data
   - Enable MongoDB authentication
   - Use IP whitelisting in MongoDB Atlas

2. **Backup:**
   - Set up automated backups in MongoDB Atlas
   - Consider point-in-time recovery for production

3. **Monitoring:**
   - Enable MongoDB Atlas monitoring
   - Set up alerts for connection failures
   - Monitor database size and performance

4. **Scaling:**
   - MongoDB Atlas auto-scales storage
   - Adjust connection pool size based on load
   - Consider read replicas for high-read workloads


