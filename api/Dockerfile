FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /Job-Description-Analysis

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /Job-Description-Analysis/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /Job-Description-Analysis

EXPOSE 8080

CMD ["python", "JobAnalyze_API.py"]