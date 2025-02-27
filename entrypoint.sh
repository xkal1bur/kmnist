#!/bin/sh

git init
git remote add origin https://github.com/xkal1bur/kmnist.git # Reemplaza con la URL de tu repositorio

echo 'Pulling Git LFS files...'
git lfs pull

echo 'Running collecstatic...'
python manage.py collectstatic --no-input --settings=kmnist.settings

echo 'Running server...'
gunicorn --env DJANGO_SETTINGS_MODULE=kmnist.settings kmnist.wsgi:application --bind 0.0.0.0:$PORT
