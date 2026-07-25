# From python:3.12.7-slim-buster
# workdir /app
# copy . /app

# Run apt update -y && apt install awscli -y

# run pip install -r requirements.txt
# cmd ["python3", "application.py"]  

FROM python:3.12.7-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python","application.py"]