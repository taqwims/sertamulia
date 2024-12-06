# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements.txt dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy aplikasi
COPY . .

# Set environment variables (ganti dengan nilai yang sesuai)
ENV GOOGLE_CLOUD_PROJECT="submissionmlgc-ahsan"
ENV MODEL_URL="https://storage.googleapis.com/penyimpanan123/model.json" 
# ... environment variables lainnya ...

# Expose port
EXPOSE 8080

# Run aplikasi
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]