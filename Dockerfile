FROM python:3.7.6-slim-buster

COPY requirements.txt /opt/program/requirements.txt

RUN pip install --no-cache-dir -r /opt/program/requirements.txt

COPY api /opt/program/code/api

ENV FLASK_APP=api
ENV FLASK_ENV=production

WORKDIR /opt/program/code/
CMD ["flask", "run", "--port", "$PORT"]