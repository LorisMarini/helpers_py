#!/bin/sh -l

set -x

# Install symbolically (editing mode)
pip install -e ${REPO_HOME_IMAGE}

# keep container alive
tail -f /dev/null
