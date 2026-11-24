FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Worker class, threads, bind and timeouts live in gunicorn.conf.py (auto-loaded).
CMD ["gunicorn", "app:app"]
