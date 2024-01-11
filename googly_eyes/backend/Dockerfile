FROM python:3.9.16-slim-bullseye
MAINTAINER Shubham Wagh "shubhamwagh48@gmail.com"

RUN apt-get update -y && \
    apt-get install -y \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 && apt-get clean

# Set the base directory
ENV BASE_DIR /app

# Set the working directory
WORKDIR $BASE_DIR

# Install poetry
RUN curl -sSL https://install.python-poetry.org | HOME=$BASE_DIR python3.9 -
# Update PATH to include Poetry's bin directory
ENV PATH "$BASE_DIR/.local/bin:${PATH}"

# Copy backend dependencies
COPY ./pyproject_backend.toml $BASE_DIR/pyproject.toml
COPY ./poetry.lock $BASE_DIR/poetry.lock

# Install backend dependencies using Poetry
RUN poetry install

# Copy backend files
COPY ./assets/filters $BASE_DIR/assets/filters
COPY ./assets/misc $BASE_DIR/assets/misc
COPY ./config $BASE_DIR/config
COPY ./googly_eyes/backend $BASE_DIR/googly_eyes/backend

# Copy backend test files
COPY ./tests/backend $BASE_DIR/tests/backend

# Run unit tests
RUN poetry run python -m unittest discover tests

# Set environment variables for the server
ENV PORT=${PORT:-8000}
ENV HOST=${HOST:-0.0.0.0}

# Expose the default port (documentation purpose)
EXPOSE $PORT

# Run the server
CMD poetry run uvicorn googly_eyes.backend.server:app --workers 4 --host $HOST --port $PORT
# CMD poetry run gunicorn googly_eyes.backend.server:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind $HOST:$PORT


