FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /Job-Description-Analysis

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /Job-Description-Analysis/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /Job-Description-Analysis

EXPOSE 5500

# CMD ["streamlit", "run", "app.py", "--server.port=5500", "--server.address=0.0.0.0"]