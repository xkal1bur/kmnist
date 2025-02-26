ARG PYTHON_VERSION=3.12-slim-bullseye

# --- Etapa Builder ---
FROM python:${PYTHON_VERSION} AS builder

# Instalar dependencias en builder (necesarias para compilar paquetes)
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip/* /tmp/requirements.txt

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# --- Etapa Final ---
FROM python:${PYTHON_VERSION}

# Instalar dependencias en etapa final (aquí es donde se ejecuta la app)
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/venv/bin:$PATH
COPY --from=builder /opt/venv /opt/venv

ARG PROJ_NAME="kmnist"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=${PROJ_NAME}.settings

WORKDIR /app
COPY . /app

# RUN python manage.py collectstatic --noinput

EXPOSE $PORT

#CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]
#CMD ["sh", "-c", "python manage.py runserver 0.0.0.0:$PORT"]
#CMD ["sh", "-c", "python manage.py runserver 0.0.0.0:${PORT:-8080}"]
CMD ["sh", "-c", "echo 'Starting server...' && python manage.py runserver 0.0.0.0:${PORT} && echo 'Server started successfully' "]
