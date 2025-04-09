#!/bin/bash
sudo apt update
sudo apt install python3-pip python3-venv nginx -y
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
gunicorn -w 4 app:app -b 0.0.0.0:8000 &