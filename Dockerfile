# 1. 베이스 이미지 선택
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. requirements.txt 먼저 복사 (캐싱 최적화)
COPY requirements.txt .

# 4. Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. 나머지 파일 복사
COPY handler.py .

# 6. 컨테이너 시작 시 실행할 명령
CMD ["python", "-u", "handler.py"]