# Use official Python image
FROM python:3.10

# Create working directory inside container
WORKDIR /app

# Copy all project files into container
COPY . .

# Install all required libraries
RUN pip install -r packages_required.txt

# Start FastAPI server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]