#!/usr/bin/env bash

python3 -m venv set_up/venv
source set_up/venv/bin/activate
pip install --upgrade pip
pip install -r set_up/requirements.txt