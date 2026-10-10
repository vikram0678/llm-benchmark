# Use an official lightweight Python runtime
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Copy dependency definition and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and project files
COPY . /app/

# Default entrypoint for running the CLI tool
CMD ["python", "src/cli.py", "--config", "config.yaml"]
