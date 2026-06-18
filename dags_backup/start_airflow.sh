#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PATH="$PROJECT_DIR/venv"

export AIRFLOW_HOME="$PROJECT_DIR"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

export JAVA_HOME=$(/usr/libexec/java_home -v 17)
source "$VENV_PATH/bin/activate"

export AIRFLOW__API__AUTH_BACKENDS="airflow.api.auth.backend.default"
export AIRFLOW__AUTH__EXPOSE_CONFIG=True
export AIRFLOW__WEB_SERVER__RBAC_SIGNUP_ROLE="Admin"
export AIRFLOW__AUTH__AUTH_ROLE_PUBLIC="Admin"
export AIRFLOW__API__AUTH_BACKENDS="airflow.api.auth.backend.basic_auth"

export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_DIR/dags"
export AIRFLOW__PYTHON_VIRTUALENV__VENV_CACHE_PATH="$PROJECT_DIR/venv_cache"

echo "Stopping old Airflow processes..."
pkill -f airflow
pkill -f java
lsof -ti:8086,8793 | xargs kill -9 2>/dev/null

rm -f "$PROJECT_DIR/airflow-webserver.pid"

echo "Starting Scheduler..."
airflow scheduler > "$PROJECT_DIR/scheduler.log" 2>&1 &

echo "Starting Webserver on port 8081..."
nohup airflow webserver --port 8081 > "$PROJECT_DIR/webserver.log" 2>&1 &

echo "---------------------------------------------------"
echo "Airflow is starting! Give it 10-15 seconds."
echo "URL: http://localhost:8081"
echo "Logs: tail -f webserver.log"
echo "To monitor logs: tail -f $PROJECT_DIR/scheduler.log"
echo "---------------------------------------------------"
