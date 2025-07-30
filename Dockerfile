# ===== BASE IMAGE =====
FROM python:3.11-slim

# ===== ENV =====
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# ===== WORKDIR =====
WORKDIR /app

# ===== COPY FILES =====
COPY . /app

# ===== INSTALL DEPENDENCIES =====
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ===== EXPOSE PORT =====
EXPOSE ${PORT}

# ===== DEFAULT CMD =====
# This runs the full pipeline (data → train → eval → serve)
CMD ["python", "main.py"]
