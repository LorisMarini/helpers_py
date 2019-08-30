Welcome to Helpers!
===============================================================================

Helpers is a collection of python 3.6 functions/classes to **move, clean, ingest,
and model** data.

## Secrets

This repo uses [transcrypt](https://github.com/elasticdog/transcrypt) to transparently encrypt/decrypt sensitive information with **aes-256-cbc** cypher (test password is `LorisMarini`). All the secrets needed to talk to the rest of the world (**Slack, Claudant, Google Sheet, Google BigQuery, S3, GCS,** ...) are stored in `/secrets`. During remote deployment in a node/cluster, the decryption key can be read from an authenticated bucket (e.g. Amazon S3). Transcrypt is included as a submodule, so to clone the repo use:

`git clone --recurse-submodules https://github.com/LorisMarini/helpers.git`

To setup transcrypt simple run

`decrypt-secrets.sh MYSECRET` replacing MYSECRET as appropriate. To see what is being encrypted, cd into the repo and type `git ls-crypt`. To change this, modify the file `.gitattributes`.

## Dependencies

The Python environment is manages via the package manager **conda**. The environment files are in `/docker`, and are divided into three sections, to make the most of docker layer caching and speed up builds

**conda_one.yml** contains all packages that passed the test of time and are always used in the base image
**conda_two.yml** contains packages that are being evaluated
**conda_local.yml** has additional packages which are useful to prototype but should not be part of the remote image

Each file is applyed to the base conda environment with commands of the type `/opt/conda/bin/conda env update -n "base" --file "/helpers/docker/conda_two.yml"` (see Dockerfile for some examples). 

## Local Deployment
For local deployments the root directory of the repo is mounted inside the image at `/helpers`, and the package installed in editing mode with `pip install -e /helpers_py`. This allows to makes changes in your favourite IDE (Pycharm) and have them immediately available in your REPL session.

1. Run `build.sh` to build the docker image locally
2. Run `exec.sh` to execute into the container. Optionally use the -r switch to

kill and recreate the containers from scratch (useful to propagate env changes). Try running a jupyter notebook at http://localhost:7040 (token is **ds**)

`./notebook.sh`

All dependencies are handled via docker-compose in the `docker-compose.yml` file. To run locally,
simply cd into the repo directory and run:

`./exec.sh -r`

The `-r` switch forces the system to kill any running containers and start fresh.
This is useful when the environment file `env-local` changes.

## Tests
The testing framework of choice is **pytest**, and all files are in `/tests`. For a
detailed breakdown of what is covered check out the index.html file in
`/tests/pytest-coverage` :D
