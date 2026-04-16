FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Docker caches this layer if requirements don't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Train the model during build (saves modello_mnist.pth inside the container)
RUN python main.py

# Expose port 5000 for the Flask server
EXPOSE 8080

# Start the Flask app when the container runs
CMD ["python", "app.py"]
