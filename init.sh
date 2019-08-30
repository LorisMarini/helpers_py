#!/bin/sh -l

set -x

# Install skepsi (symbolically)
pip install -e ${SKEPSI_HOME_IMAGE}

# Set creds to login in Airflow webserver
# python /skepsi/airflow/login/set_creds.py

# Make a local dir to simulate external disk mounted in remote environment
python /skepsi/airflow/dirs/consolidate.py

airflow initdb
tail -f /dev/null

# airflow webserver -D
# airflow scheduler
