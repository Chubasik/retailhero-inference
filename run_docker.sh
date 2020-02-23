#!/bin/bash
set -e

/usr/bin/tf_serving_entrypoint.sh & gunicorn --bind 0.0.0.0:8000 server:app