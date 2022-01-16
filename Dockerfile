# syntax=docker/dockerfile:1
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install rlcard[torch]==1.0.5

RUN pip3 install -r requirements.txt

COPY . .

ENV PYTHONPATH /app

ENTRYPOINT [ "python", "examples/human/run_wizard_human_trickpred.py"]