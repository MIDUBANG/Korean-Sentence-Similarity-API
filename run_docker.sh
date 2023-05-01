#!/bin/bash
docker-compose stop flask_app
docker-compose stop nginx 
docker-compose rm

echo "start docker-compose up: ubuntu"
sudo docker-compose -f /home/ubuntu/srv/nlp_clova/docker-compose.yml up --build -d 