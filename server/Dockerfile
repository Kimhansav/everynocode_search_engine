FROM python:3.11.2
WORKDIR /app

# 서비스 계정 키 파일을 /secrets 디렉터리에 복사
COPY bubble-project-420201-ee79783bde67.json /secrets/bubble-project-420201-ee79783bde67.json

# 환경 변수 설정
ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/bubble-project-420201-ee79783bde67.json

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "--log-level", "debug", "--timeout", "1000", "-w 4", "-b", "0.0.0.0:8000", "app:create_app()"]