FROM python:3.10
WORKDIR /app
COPY requirements.txt .
COPY wheels ./wheels
RUN pip install --no-index --find-links=/app/wheels -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["python","-m","uvicorn","src.api.app:app","--host","0.0.0.0","--port","8000"]
