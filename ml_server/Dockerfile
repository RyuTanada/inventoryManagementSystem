# Use official Python image
FROM python:3.10

# Set working directory inside container
WORKDIR /ml_server

# Copy everything inside ./ml_server folder into the container
COPY . .

# Install required Python libraries
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn pandas scikit-learn tensorflow

# Expose FastAPI default port
EXPOSE 8001

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]