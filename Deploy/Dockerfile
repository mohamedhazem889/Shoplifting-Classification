FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p media/uploads

EXPOSE 8000

CMD ["gunicorn", "shoplifting_api.wsgi:application", \
     "--workers", "2", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:8000"]
