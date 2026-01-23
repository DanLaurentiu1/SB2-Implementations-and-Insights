FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root

ENV PYTHONPATH=/app

COPY . .

RUN chmod +x entrypoint.sh
RUN sed -i 's/\r$//' ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]