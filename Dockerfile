FROM python:3.11.2
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "--timeout", "1000", "-w 4", "-b", "0.0.0.0:8000", "app.wsgi:create_app"]