#!/bin/bash
jupyter lab \
	--notebook-dir="$JUPYTER_HOME_IMAGE" \
	--no-browser \
  --allow-root \
	--port=8889 \
	--ip=0.0.0.0 \
	--LabApp.token='ds'
