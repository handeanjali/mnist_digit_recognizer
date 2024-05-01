# Stage 1: Build stage
FROM python:3.9-slim AS build

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY app.py mnist_model.keras requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.9-slim AS runtime

# Set working directory
WORKDIR /app

# Copy files from the build stage
COPY --from=build /app .

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
