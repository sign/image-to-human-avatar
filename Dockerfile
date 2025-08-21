FROM python:3.12-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

## Install system dependencies
RUN apt-get update

# ffmpeg: a dependency of mediapipe
RUN apt-get install -y ffmpeg

# Setup local workdir and dependencies
WORKDIR /app

# Install torch for CPU only, since the model runs faster on CPU, and this results in and smaller docker image
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other python dependencies.
ADD ./pyproject.toml ./pyproject.toml
RUN mkdir -p human_avatar && touch README.md
RUN pip install --no-cache-dir ".[server]"

# Copy local code to the container image.
COPY ./human_avatar ./human_avatar

# Copy assets to the container image, for priming
COPY ./assets ./assets

# Prime the cache by downloading models
RUN python -m human_avatar.example

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 human_avatar.server:app
