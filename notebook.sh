#!/bin/bash
jupyter notebook \
	--notebook-dir="$JUPYTER_HOME_IMAGE" \
	--no-browser \
	--allow-root \
	--port=7740 \
	--ip=0.0.0.0 \
	--NotebookApp.token='ds'
