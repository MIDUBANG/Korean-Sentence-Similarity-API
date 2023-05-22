#!/bin/bash

echo "start docker-compose up: ubuntu"
sudo docker-compose -f /home/ubuntu/srv/nlp_clova/docker-compose.yml up --build -d
cd /home/ubuntu/srv/nlp_clova/flask_app
celery -A app.celery worker --loglevel=info
