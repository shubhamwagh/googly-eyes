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

# Copy frontend dependencies
COPY ./pyproject_frontend.toml $BASE_DIR/pyproject.toml
COPY ./poetry.lock $BASE_DIR/poetry.lock

# Install dependencies using Poetry
RUN poetry install

# Copy frontend files
COPY ./googly_eyes/frontend $BASE_DIR/googly_eyes/frontend

# Copy frontend test files
COPY ./tests/frontend $BASE_DIR/tests/frontend

# Run unit tests
RUN poetry run python -m unittest discover tests

# Set environment variables for the server
ENV PORT=${PORT:-8501}
ENV HOST=${HOST:-0.0.0.0}
ENV URL_BACKEND=${URL_BACKEND:-http://backend:8000/}

# Expose the default port (documentation purpose)
EXPOSE $PORT

# Run the viewer
WORKDIR $BASE_DIR/googly_eyes/frontend
CMD poetry run streamlit run view.py --server.address $HOST --server.port $PORT --server.headless true

