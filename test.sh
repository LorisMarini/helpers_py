#!/usr/bin/env bash

set -ve

# Set location of creds to decrypt
TRANSCRYPT_KEY="s3://aphq-secrets/buildkite/secrets/skepsi-transcrypt-key.txt"
SECRETS_DIR="$(pwd)/secrets"

# decrypt the secrets
./transcrypt/transcrypt --flush-credentials --force --yes || :
./transcrypt/transcrypt -c aes-256-cbc -p "$(aws s3 cp ${TRANSCRYPT_KEY} -)" --force --yes


function tests() {
  local ENVIRONMENT=$1
  local ENV_FILE="./secrets/env-${ENVIRONMENT}"
  local CONTAINER_NAME=skepsi-pytest-$ENVIRONMENT
  echo "$ENVIRONMENT: Running image with env file ${ENV_FILE}"
  docker rm -f $CONTAINER_NAME || :
  docker run \
    --name $CONTAINER_NAME \
    -v $SECRETS_DIR:"/skepsi/secrets" \
    --env-file ./${ENV_FILE} \
    "${IMAGE}" \
    /bin/sh -l -c "/opt/conda/bin/pytest \
      -v \
      --disable-pytest-warnings \
      --cov='/skepsi' \
      --cov-report html:/skepsi/tests/pytest-coverage \
      --basetemp='/skepsi/tests/pytest-basetemp' '/skepsi'" > "/tmp/skepsi-$ENVIRONMENT-tests.txt"
}

mapfile -t ENVIRONMENTS < <(buildkite-agent meta-data get environment)

for ENVIRONMENT in "${ENVIRONMENTS[@]}"; do
  echo "$ENVIRONMENT: Starting tests"
  tests $ENVIRONMENT &
done

wait

for ENVIRONMENT in "${ENVIRONMENTS[@]}"; do
  # show test results
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "$ENVIRONMENT: Test results"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  cat "/tmp/skepsi-$ENVIRONMENT-tests.txt"

  # see if tests failed
  if cat "/tmp/skepsi-$ENVIRONMENT-tests.txt" | grep error; then
    echo "Test failed in $ENVIRONMENT"
    FAILED=1
  fi

  # if last item in array, grab the test coverage
  if [[ $ENVIRONMENT == "${ENVIRONMENTS[-1]}" ]]; then
    ## generate badges for passing tests
    # # pytest coverage
    docker cp skepsi-pytest-$ENVIRONMENT:/skepsi/tests/pytest-coverage/index.html pytest.html
    (sed -n 's|.*<span class="pc_cov">\(.*\)</span>.*|\1|p' pytest.html) | buildkite-agent meta-data set pytest_coverage
  fi
done

# set stickers/badges
if [[ $FAILED == 1 ]]; then
  # tests are failing
  buildkite-agent meta-data set test_results 'failing'
  exit 1
else
  # tests are passing
  buildkite-agent meta-data set test_results "passing"
fi
