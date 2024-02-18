#
# Multi Stage: Builder Image
#
FROM python:3.10-slim AS builder

# Install poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1
RUN pip install poetry

# Make working directory
RUN mkdir -p /app

# Copy necessary files
COPY ./pyproject.toml /app
COPY ./poetry.lock /app
COPY ./README.md /app
RUN mkdir -p /app/data/01_raw
COPY ./data/01_raw /app/data/01_raw
RUN ls -la /app/data/*
COPY ./scripts /app/scripts
COPY ./src /app/src

# Set working directory
WORKDIR /app

# Install python dependencies in container
RUN poetry install --without dev,vis

#
# Multi Stage: Runtime Image
#
FROM python:3.10-slim AS runtime

# Copy over baked environment
COPY --from=builder /app /app

# Set 
WORKDIR /app

# Set executables in PATH
ENV PATH="/app/.venv/bin:$PATH"

# Run application code
CMD ["/app/scripts/project_entry_point.sh"]
