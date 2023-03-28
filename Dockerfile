FROM python:3.10-slim

WORKDIR /app

COPY dist/decipher-0.1.12-py3-none-any.whl .
RUN --mount=type=cache,target=/root/.cache pip install decipher-0.1.12-py3-none-any.whl

COPY examples examples
