FROM python:3.12-slim

RUN adduser app
USER app
WORKDIR /app

RUN pip install --upgrade pip

ADD ./requirements.txt /app/
RUN pip install --no-cache-dir --upgrade --user -r requirements.txt

ADD ./server/ /app/server/

CMD ["python", "-m", "fastapi", "run", "server/api.py", "--port", "5000"]
