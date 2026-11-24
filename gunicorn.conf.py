# Auto-loaded by Gunicorn from the working directory (works for both the
# Dockerfile CMD and a plain `gunicorn app:app` Render start command).
#
# Why: the default sync worker handles ONE request at a time. With a single
# worker (512MB free tier), any slow request — ImgBB upload in the n8n
# callback (up to 20s), n8n/Supabase calls, a heavy feed — froze the whole
# API for every other user. Threads let I/O-bound requests overlap while
# staying in one process (so in-memory caches + rate limiter stay coherent
# and RAM stays flat).
import os

bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = int(os.getenv('WEB_CONCURRENCY', '1'))
worker_class = 'gthread'
threads = int(os.getenv('GUNICORN_THREADS', '8'))

# Slow upstreams (n8n, ImgBB) can legitimately take a while; don't kill the
# worker mid-request at the 30s default.
timeout = int(os.getenv('GUNICORN_TIMEOUT', '120'))
graceful_timeout = 30
keepalive = 5

accesslog = '-'
errorlog = '-'
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')
