#!/bin/bash
# Fires Jupyter Lab inside the local docker container.
# Port is 8889. Mapping is defined in docker-compose.yml
jupyter lab \
	--notebook-dir="$JUPYTER_HOME_IMAGE" \
	--no-browser \
  --allow-root \
	--port=8889 \
	--ip=0.0.0.0 \
	--LabApp.token='ds'
