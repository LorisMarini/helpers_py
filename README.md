Welcome to Helpers!
===============================================================================

Helpers is a collection of python 3.6 functions/classes to **move, clean, ingest,
and model** data.

## Secrets

This repo uses [transcrypt](https://github.com/elasticdog/transcrypt) to transparently
encrypt/decrypt sensitive information. All secrets used to talk to the rest of
the world (**Slack, Claudant, Google Sheet, Google BigQuery, S3, GCS,** ...) are
stored in `/secrets`. During remote deployment in a node/cluster, the decryption
key can be read from an authenticated bucket (e.g. Amazon S3). Transcrypt is included
as a submodule, so to clone the repo use:

`git clone --recurse-submodules https://github.com/LorisMarini/helpers_py.git`

To setup transcrypt simple run

`decrypt-secrets.sh MYSECRET` replacing MYSECRET as appropriate.

## Local Deployment
For local deployments the root directory of the repo is mounted inside the image
at `/helpers_py`, and the package installed in editing mode with `pip install -e /helpers_py`.
This allows to makes changes in your favourite IDE (Pycharm) and have them immediately
available in your REPL session.

1. Run `build.sh` to build the docker image locally
2. Run `exec.sh` to execute into the container. Optionally use the -r switch to
kill and recreate the containers from scratch (useful to propagate env changes).
Try running a jupyter notebook with with:

`./jupyter.sh`

the notebook UI appears at http://localhost:7040, token is 'ds'. When deployed
locally airflow uses a local postgreSQL database and the localExecutor. All dependencies
are handled via docker-compose in the `docker-compose.yml` file. To run locally,
simply cd into the repo directory and run:

`./exec.sh -r`

The `-r` switch forces the system to kill any running containers and start fresh.
This is useful when the environment file `env-local` changes.

## Tests
The testing framework of choice is **pytest**, and all files are in `/tests`. For a
detailed breakdown of what is covered check out the index.html file in
`/tests/pytest-coverage` :D
