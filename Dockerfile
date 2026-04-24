FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir gymnasium minigrid fastapi uvicorn requests pydantic

COPY src/ ./src/
