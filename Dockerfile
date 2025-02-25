ARG PYTHON_VERSION=3.12-slim-bullseye
FROM python:${PYTHON_VERSION} AS builder

# Combine venv creation
RUN python -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

# Install packages and cleanup in same layer
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip/* /tmp/requirements.txt

# Final stage
FROM python:${PYTHON_VERSION}

ENV PATH=/opt/venv/bin:$PATH

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

ARG PROJ_NAME="kmnist"
# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=${PROJ_NAME}.settings

WORKDIR /app
COPY . /app

RUN python manage.py collectstatic --noinput

EXPOSE 8080

CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]
