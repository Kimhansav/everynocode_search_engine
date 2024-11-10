FROM python:3.11.2
WORKDIR /search-tutorial

# 서비스 계정 키 파일을 /secrets 디렉터리에 복사
COPY fiery-cairn-436011-p1-5fec1076b526.json /secrets/fiery-cairn-436011-p1-5fec1076b526.json

# 환경 변수 설정
ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/fiery-cairn-436011-p1-5fec1076b526.json

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# RUN flask reindex

EXPOSE 8000
CMD ["gunicorn", "--log-level", "debug", "--timeout", "1000", "-w 4", "-b", "0.0.0.0:8000", "app:app"]


