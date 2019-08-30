#!/usr/bin/env bash

# REMOTE BUILDS
# See https://cloud.google.com/container-registry/docs/advanced-authentication#advanced_authentication_methods
# for a description of the authentitaction method used. A file called "ap-data-science-1-5b843be43044.json"
# was manually exported from GCR and loaded to the bucket s3://aphq-secrets/ which can be accessed
# by the buildkite agent via IAM roles in AWS.

if [ -z "$BUILDKITE" ]; then
  # LOCAL BUILD
  echo "Building image locally"
  docker-compose build
else
  # BUILDKITE BUILD
  echo "Starting build image $IMAGE on Buildkite"

  # flush transcrypt secrets
  ./transcrypt/transcrypt --flush-credentials --force --yes || :

  # Authenticate to GCR
  echo "Logging into $REGISTRY_HOST to pull cache images"
  aws s3 cp $GCP_KEY - | docker login -u _json_key --password-stdin https://"${REGISTRY_HOST}"

  echo "Pulling cache from branch if exists, otherwise use master branch image as cache"
  if docker pull "$IMAGE"; then
    CACHE=$IMAGE
  elif docker pull ${COMMIT_IMAGE}; then
    CACHE=${COMMIT_IMAGE}
  else
    MASTER_CACHE="${REGISTRY_HOST}/${REGISTRY_REPO}:master"
    docker pull "$MASTER_CACHE"
    CACHE=$MASTER_CACHE
  fi

  echo "Building Docker image"
  docker build --compress \
  -f Dockerfile.remote \
  --cache-from "${CACHE}" \
  --tag "${IMAGE}" \
  .

  docker tag ${IMAGE} "${REGISTRY_HOST}/${REGISTRY_REPO}:${BUILDKITE_COMMIT}"
  docker push "${IMAGE}"
  docker push "${REGISTRY_HOST}/${REGISTRY_REPO}:${BUILDKITE_COMMIT}"
fi
