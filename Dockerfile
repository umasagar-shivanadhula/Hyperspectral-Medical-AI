
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure project root is on PYTHONPATH so all absolute imports resolve
ENV PYTHONPATH=/app

# Create required output and log directories so the container starts cleanly
RUN mkdir -p outputs/heatmaps outputs/feature_importance outputs/spectral_analysis \
             outputs/model_comparison outputs/trained_models outputs/evaluation_metrics \
             outputs/prediction_results logs/audit datasets

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
