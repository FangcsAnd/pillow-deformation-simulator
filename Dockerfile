FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi uvicorn python-multipart numpy scipy trimesh rtree

COPY backend/ ./backend/
COPY frontend/ ./frontend/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
