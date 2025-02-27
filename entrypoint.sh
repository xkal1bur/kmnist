#!/bin/sh
set -x

git init main

echo 'Adding remote repository...'
git remote add origin https://github.com/xkal1bur/kmnist.git
git remote -v
git fetch

git lfs env

echo 'Pulling Git LFS files...'
git lfs pull

ls -la

echo 'Running collecstatic...'
python manage.py collectstatic --no-input --settings=kmnist.settings

echo 'Running server...'
gunicorn --env DJANGO_SETTINGS_MODULE=kmnist.settings kmnist.wsgi:application --bind 0.0.0.0:$PORT
