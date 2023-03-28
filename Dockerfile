FROM python:3.10-slim

WORKDIR /app

COPY dist/decipher-0.1.13-py3-none-any.whl .
RUN --mount=type=cache,target=/root/.cache pip install decipher-0.1.13-py3-none-any.whl[pyarrow]

COPY examples examples
