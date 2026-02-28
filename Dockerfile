FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cache layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . 2>/dev/null || \
    pip install --no-cache-dir \
        "pandas>=2.2,<3" \
        "numpy>=1.26" \
        "apscheduler>=3.10,<4" \
        "scikit-learn>=1.4,<2" \
        "joblib>=1.3" \
        "fastapi>=0.110,<1" \
        "uvicorn[standard]>=0.29,<1" \
        "streamlit>=1.35,<2" \
        "plotly>=5.22,<6" \
        "python-dotenv>=1.0" \
        "loguru>=0.7"

# Copy application code
COPY src/ src/
COPY seed/ seed/
COPY dashboard/ dashboard/
COPY models/ models/
COPY docs/ docs/
COPY pyproject.toml .
# Create data directories
RUN mkdir -p data/raw data/processed

# Install the project in editable mode
RUN pip install --no-cache-dir -e .

# Default: run the API
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
