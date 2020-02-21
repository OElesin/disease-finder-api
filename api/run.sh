#!/usr/bin/env bash

EXEC_PORT=${1//[^0-9]/}

echo "Passed Variable: ${1}"
echo "Running Port: ${EXEC_PORT}"

flask run --port ${EXEC_PORT}