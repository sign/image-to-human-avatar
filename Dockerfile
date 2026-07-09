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

# Prime the cache by downloading models.
# briaai/RMBG-2.0 is a gated repo, so the download needs an authenticated HF_TOKEN;
# a build secret keeps the token out of the image layers.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) python -m human_avatar.example

# All models are baked in above; offline mode avoids startup HEAD requests to
# huggingface.co, which fail with 401 for the gated RMBG-2.0 repo
ENV HF_HUB_OFFLINE=1

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 human_avatar.server:app
