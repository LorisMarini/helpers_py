#!/usr/bin/env bash

# Source env file
set -a
source ./secrets/env-local
# shellcheck disable=SC1090
set +a

# Name of the repo
REPOSITORY_BASE_NAME="helpers"

function countdown() {
  secs=$1
  shift
  msg=$*
  while [ $secs -gt 0 ]; do
    printf "$msg%.d" $((secs--))
    sleep 1
  done
  echo
}
function countdown_info() { countdown $1 "\r\E[0;34mINFO   : $2"; }
function countdown_error() { countdown $1 "\r\E[0;31mERROR  : $2"; }
function countdown_warning() { countdown $1 "\r\E[0;33mWARNING: $2" ;}

# color echo = cecho
cecho () {
  declare -A colors;
  colors=(\
      ['red']='\E[0;31m'\
      ['yellow']='\E[0;33m'\
      ['blue']='\E[0;34m'\
  );
  local defaultMSG="No message passed.";
  local defaultColor="blue";
  local defaultNewLine=true;
  while [[ $# -gt 1 ]]; do
    key="$1";
    case $key in
        -c|--color)
            color="$2";
            shift;
        ;;
        -n|--noline)
            newLine=false;
        ;;
        *)
            # unknown option
        ;;
    esac
    shift;
  done
  message=${1:-$defaultMSG};   # Defaults to default message.
  color=${color:-$defaultColor};   # Defaults to default color, if not specified.
  newLine=${newLine:-$defaultNewLine};
  echo -en "${colors[$color]}";
  echo -en "$message";
  if [ "$newLine" = true ]; then
    echo;
  fi
  tput sgr0; #  Reset text attributes to normal without clearing screen.
  return;
}
function warning { cecho -c 'yellow' "WARNING: $*"; }
function error { cecho -c 'red' "ERROR  : $*"; }
function info { cecho -c 'blue' "INFO   : $*"; }

function delete_container { docker-compose rm -f -s; }

function run_container {
  info "Creating new container, mounting ${JUPYTER_HOME_LOCAL} to ${JUPYTER_HOME_IMAGE}"
  set -e
  docker-compose up -d
  set +e
  exec_tasks
}

function exec_tasks {
  if [[ $root == 1 ]]; then
    info "Exec'ing into container"
    docker-compose exec -u 0 $REPOSITORY_BASE_NAME sh -l
  elif [[ $logging == 1 ]]; then
    info "Grabbing docker-compose logs"
    docker-compose logs -f
  else
    if [[ $test == 1 ]]; then
      info "Running tests"
      docker-compose exec $REPOSITORY_BASE_NAME sh -l -c \
      "/opt/conda/bin/pytest \
      -v \
      --cov=/helpers \
      --cov-config=/helpers/.coveragerc \
      --cov-report html:/helpers/tests/pytest-coverage\
      --disable-pytest-warnings \
      --basetemp=/helpers/tests/pytest-basetemp /helpers"
    else
      docker-compose exec $REPOSITORY_BASE_NAME sh -l
    fi
  fi
  exit 0
}

function check_for_running_container {
  if [[ "$(docker-compose ps --filter "status=running" --services | grep $REPOSITORY_BASE_NAME)" ]]; then
    info "Found running container"
    exec_tasks
  fi
}

function main {
  # If container exists and is running just exec into it
  check_for_running_container
  # if not running, then run it
  run_container
}

# If user passes -v switch, set -x will improve verbosity in case of errors
# with the -r switch we recreate the container *(no work preservation)
while getopts ":rbvstl" opt; do
  case $opt in
    v)
      set -x;
      ;;
    b)
      ./build.sh
      ;;
    r)
      # Delete and recreate container
      warning "Deleting and recreating container";
      delete_container;
      ;;
    s)
    # Exec as root
      root=1
      ;;
    t)
    # Run all tests
      test=1
      ;;
    l)
    # show logs
      logging=1
      ;;
    \?)
      error "'-$OPTARG' arg doesn't exist.";
      exit 1
      ;;
  esac
done

# If no arguments passed to the script just execute the main
main
