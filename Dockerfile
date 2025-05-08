FROM python:3.10

WORKDIR /app

# Install system dependencies needed for dlib
RUN apt-get update && apt-get install -y cmake build-essential

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
