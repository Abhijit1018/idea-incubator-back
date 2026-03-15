# Idea Incubator Backend

Flask backend for the Idea Incubator platform.

## What It Does

- Accepts idea/tool submissions from frontend.
- Triggers n8n workflows for enrichment.
- Stores generated catalogs in SQL database.
- Receives webhook callbacks from n8n.
- Supports chat history per catalog entry.
- Supports optional semantic search with SentenceTransformers + Pinecone.

## Tech Stack

- Python 3.13
- Flask + Flask-Cors
- SQLAlchemy
- SQLite (local) or PostgreSQL (production)
- Optional: Pinecone vector index + sentence-transformers embeddings

## API Endpoints

- `POST /api/ideas/submit`
- `GET /api/catalogs/`
- `POST /api/catalogs/search`
- `POST /api/webhooks/n8n-callback`
- `POST /api/webhooks/chat-message`
- `GET /api/catalogs/<entry_id>/chat`
- `POST /api/catalogs/<entry_id>/chat`
- `POST /api/catalogs/<entry_id>/generate-embedding`

## Local Development

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment:

```bash
cp .env.example .env
```

3. Run backend:

```bash
python app.py
```

Backend runs on `http://localhost:5000` by default.

## Environment Variables

See `.env.example` for all options. Key variables:

- `DATABASE_URL`
- `BACKEND_BASE_URL`
- `N8N_WEBHOOK_URL`
- `N8N_CHAT_WEBHOOK_URL`
- `PINECONE_API_KEY` (optional)

If Pinecone is not configured, vector-index operations are skipped and fallback text search is used where possible.

## Docker Deployment

Build image:

```bash
docker build -t idea-incubator-backend .
```

Run container:

```bash
docker run --env-file .env -p 5000:5000 idea-incubator-backend
```

## Production Notes

- Use PostgreSQL via `DATABASE_URL`.
- Set `BACKEND_BASE_URL` to your public backend URL so uploaded image URLs are correct.
- Ensure n8n can call `POST /api/webhooks/n8n-callback`.
- For persistent uploads, mount `static/uploads` to durable storage.
