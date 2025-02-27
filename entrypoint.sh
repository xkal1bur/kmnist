#!/bin/sh

git config --global init.defaultBranch main
git init

echo 'Adding remote repository...'
git remote add origin https://github.com/xkal1bur/kmnist.git

echo 'Pulling Git LFS files...'
git pull origin main --strategy=ours

ls -la recognition/

echo 'Running collecstatic...'
python manage.py collectstatic --no-input --settings=kmnist.settings

echo 'Running server...'
gunicorn --env DJANGO_SETTINGS_MODULE=kmnist.settings kmnist.wsgi:application --bind 0.0.0.0:$PORT
