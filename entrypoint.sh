#!/bin/sh

echo 'Running collecstatic...'
python manage.py collectstatic --no-input --settings=config.settings.production

echo 'Running server...'
gunicorn --env DJANGO_SETTINGS_MODULE=config.settings.production config.wsgi:application --bind 0.0.0.0:$PORT
