FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


# Create directories for feedback data
RUN mkdir -p /app/feedback_data/images
RUN mkdir -p /app/feedback_data/labels
RUN mkdir -p /app/models
COPY best_model.pth /app/models

# Create an empty __init__.py file in the root directory to make it a package
RUN touch __init__.py
RUN touch utils/__init__.py

# Set PYTHONPATH to include the current directory
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "prediction:app", "--host", "0.0.0.0", "--port", "8000"]