FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD uvicorn feature_extraction:app --port=8000 --host=0.0.0.0
