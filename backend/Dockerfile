FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY filinglens filinglens
COPY eval eval
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "filinglens.api:app", "--host", "0.0.0.0", "--port", "8000"]
