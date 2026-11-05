
FROM python:3.13-slim

WORKDIR /app

# Install project dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run the complete MLOps pipeline
CMD ["python", "src/mlops_pipeline/chocolate_prediction_pipeline.py"]
