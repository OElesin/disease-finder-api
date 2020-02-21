#!/usr/bin/env bash


echo "Passed Variable: ${PORT}"
EXEC_PORT=${PORT//[^0-9]/}

echo "Running Port: ${EXEC_PORT}"

flask run --port ${EXEC_PORT}