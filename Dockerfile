# ─────────────────────────────────────────────────────────
# Dockerfile — Crop Advisor AI
# Build: docker build -t anitharajan/crop-advisor-ai:v1 .
# Run:   docker run -p 5000:5000 -e GEMINI_API_KEY=your_key anitharajan/crop-advisor-ai:v1
# ─────────────────────────────────────────────────────────
FROM python:3.10-slim

LABEL maintainer="anitharajan"
LABEL description="Crop Recommendation AI - Random Forest + Gemini LLM"
LABEL version="1.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/ templates/
COPY static/ static/
COPY model/ model/

RUN mkdir -p logs

ENV PORT=5000
ENV GEMINI_API_KEY=your_key_here

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["gunicorn","--bind","0.0.0.0:5000","--workers","2","--timeout","120","app:app"]
