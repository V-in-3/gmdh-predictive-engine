#!/bin/bash

PROJECT_DIR="./gmdh-predictive-engine"
export AIRFLOW_HOME="$PROJECT_DIR"

export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_DIR/dags"
export JAVA_HOME=$(/usr/libexec/java_home -v 17)

echo "Stopping old Airflow processes..."
pkill -f airflow
lsof -ti:8086,8793 | xargs kill -9 2>/dev/null

echo "Starting Scheduler..."
airflow scheduler > "$PROJECT_DIR/scheduler.log" 2>&1 &

echo "Starting Webserver on port 8086..."
airflow webserver --port 8086 > "$PROJECT_DIR/webserver.log" 2>&1 &

echo "---------------------------------------------------"
echo "Airflow is starting! Give it 10-15 seconds."
echo "URL: http://localhost:8086"
echo "Logs: tail -f webserver.log"
echo "---------------------------------------------------"