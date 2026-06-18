from __future__ import annotations

import base64
import gzip
import io
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import boto3
import pandas as pd
import requests
from airflow.decorators import dag
from airflow.decorators import task
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonVirtualenvOperator, PythonOperator
from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook
from airflow.providers.amazon.aws.operators.s3 import S3DeleteObjectsOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule
from cogent.operators.athena import AthenaQueryOperator
from cogent.utils import airflow_callbacks
from cogent.utils.variables import AWS_ENV, AWS_ACCOUNT_ID


# If running locally, use project path; otherwise use standard /opt
PROJECT_ROOT = os.getenv("AIRFLOW_HOME", "/opt/airflow")

# For venv cache: use a folder inside the project if on Mac
if "/opt/airflow" not in PROJECT_ROOT and os.path.exists(os.path.join(PROJECT_ROOT, "shared")):
    SHARED_ROOT = f"{PROJECT_ROOT}/shared"
else:
    SHARED_ROOT = "/opt/airflow_share"

VENV_BASE_DIR = f"{SHARED_ROOT}/cogent_venv"

Path(VENV_BASE_DIR).mkdir(parents=True, exist_ok=True)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

CURRENT_PYTHONPATH = os.environ.get('PYTHONPATH', '')
NEW_PYTHONPATH = f"{PROJECT_ROOT}:{CURRENT_PYTHONPATH}"

from jobs.airflow_pylint_qa.regression import get_historical_scores
from jobs.airflow_pylint_qa.utils import dags_to_analyze_from_config

# Configuration constants
DAG_NAME = 'airflow_pylint_qa'
AIRFLOW_API_CONN_ID = "airflow_api_connection"
SLACK_WEBHOOK_CONN_ID = "slack_default"

DATABASE = "airflow_validation"
TABLE = "results"
INTEGRATION_TEST_BUCKET = f"edm-datalake-integration-test-{AWS_ACCOUNT_ID}"
ICEBERG_TABLE_KEY = "qa/pylint"
ICEBERG_TABLE_LOCATION = f"s3://{INTEGRATION_TEST_BUCKET}/{ICEBERG_TABLE_KEY}"
OUTPUT_LOCATION = f"s3://{INTEGRATION_TEST_BUCKET}/qa/query"

WORK_GROUP = "edm"
TARGET_ENVS = ["L1", "L2", "L3"]

DEFAULT_ARGS = {
    "owner": "airflow",
    "provide_context": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=6),
    "on_success_callback": None,
    # lambda ctx: airflow_callbacks.success(
    #     context=ctx, pipeline_name=DAG_NAME
    # ),
    "on_failure_callback": None,
    # lambda ctx: airflow_callbacks.failed(
    #     context=ctx, pipeline_name=DAG_NAME
    # ),
}

# ==========================================================
# 0. LOGGING
# ==========================================================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Creating a logger for the DAG file
LOG = logging.getLogger(__name__)
# Setting DEBUG level for this file
LOG.setLevel(logging.DEBUG)

# stdout or stderr
LOG_TYPE = "stderr"
# New keywords for improved Airflow log filtering
AIRFLOW_ERROR_KEYWORDS = ["error", "exception", "traceback", "attributeerror", "typeerror", "valueerror",
                          "failed to execute", "failed"]
# Maximum log length for display in Slack (to prevent overflow)
MAX_SLACK_LOG_LENGTH = 1500

# --- Quality check constant ---
# Pylint: Set the minimum acceptable score for the entire project.
PYLINT_QUALITY_TARGET = 5.0

# --- Batching Constant for Athena Fix ---
# Maximum number of Pylint issues (rows) to include in a single INSERT INTO SQL query.
# Set low (100) to safely prevent the 256KB 'InvalidRequestException' limit.
ATHENA_INSERT_BATCH_SIZE = 100

ATHENA_N_MINUS_1_RESULTS_SQL_TEMPLATE = """
    WITH ranked_scores AS (
        SELECT 
            dag_id,
            CAST(check_timestamp AS timestamp) AS check_timestamp,
            overall_score,
            issue_type,
            symbol,
            symbol_code,
            message,
            line_number,
            result_of_last_run,
            CAST(time_of_last_run AS timestamp) AS time_of_last_run ,
            ROW_NUMBER() OVER (
                PARTITION BY dag_id 
                ORDER BY check_timestamp DESC
            ) as rn
        FROM {database}.{table} 
        WHERE {where_filter}
    )
    SELECT 
        dag_id,
        CAST(check_timestamp AS timestamp) AS check_timestamp,
        overall_score,
        issue_type,
        symbol,
        symbol_code,
        message,
        line_number,
        result_of_last_run,
        CAST(time_of_last_run AS timestamp) AS time_of_last_run 
    FROM ranked_scores
    WHERE rn = 2
"""


def get_airflow_base_url():
    from cogent.utils.variables import AWS_ENV

    # return f"https://edm-cogent-airflow.example{AWS_ENV}.int"
    return f"http://localhost:8081"


class EmrLogParser:
    """Parses the Airflow EMR log string to extract S3 and EMR identifiers."""

    def __init__(self, log_output: str):
        self.log_output = log_output
        self.log_file_line: Optional[str] = None
        self.s3_log_bucket: Optional[str] = None
        self.aws_account_id: Optional[str] = None
        self.cluster_id: Optional[str] = None
        self.step_id: Optional[str] = None

    def find_log_line(self) -> bool:
        """Searches for the 'LogFile:' line in the provided Airflow log."""
        for line in self.log_output.splitlines():
            if "LogFile: s3://" in line:
                self.log_file_line = line
                return True
        return False

    def parse_log_url(self) -> bool:
        """Parses the S3 URL to extract AWS Account ID, CLUSTER_ID, and STEP_ID."""
        if not self.log_file_line:
            return False

        log_url = self.log_file_line.split("LogFile: ")[-1].strip()

        # Groups: 1-BucketName, 2-AccountID, 3-Region, 4-ClusterID, 5-StepID
        match = re.search(
            r's3://(acom-edm-cogent-emrcluster-(\d+)-([a-z]{2}-[a-z]+-[1-9]))/emr_logs/(j-[\w\d]+)/steps/(s-[\w\d]+)/',
            log_url)

        if match:
            self.s3_log_bucket = match.group(1)
            self.aws_account_id = match.group(2)
            self.cluster_id = match.group(4)
            self.step_id = match.group(5)
            return True
        return False

    def get_params(self) -> Optional[Tuple[str, str, str]]:
        """Performs the full parsing process and returns the parameters."""
        if self.find_log_line() and self.parse_log_url():
            return (self.s3_log_bucket, self.cluster_id, self.step_id)
        return None


def get_emr_log_content_from_s3(bucket_name, cluster_id, step_id, log_type):
    """
    Downloads and returns the content of a gzipped EMR step log file directly from S3.
    """
    s3_key = f"emr_logs/{cluster_id}/steps/{step_id}/{log_type}.gz"

    LOG.info(f"   EMR Log Link found. Trying to download S3://{bucket_name}/{s3_key}")

    try:
        session = boto3.Session(region_name="us-east-1")
        s3_client = session.client('s3')
    except Exception as e:
        return f"❌ EMR Error: Failed to initialize Boto3 session or S3 client (Check Airflow Env profile): {e}"

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        gzipped_content = response['Body'].read()

        with gzip.GzipFile(fileobj=io.BytesIO(gzipped_content)) as gz_file:
            log_content = gz_file.read().decode('utf-8')

        return log_content

    except s3_client.exceptions.NoSuchKey:
        return f"❌ EMR Error: The specified log file does not exist on S3: {s3_key}"
    except Exception as e:
        return f"❌ EMR Error: An unexpected error occurred during S3 download: {e}"


def get_all_active_dags(base_url, auth_header=None) -> Dict[str, bool]:
    """
    Fetches a list of all active DAG IDs from the Airflow server using pagination.
    """
    all_dag_ids = {}
    limit = 100
    offset = 0
    total_entries = None

    headers = {"Content-Type": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header

    LOG.info("\n🔄 Fetching all active DAGs...")

    try:
        while total_entries is None or offset < total_entries:
            url = f"{base_url}/api/v1/dags"
            params = {
                "limit": limit,
                "offset": offset,
                "only_active": True
            }

            response = requests.get(url, headers=headers, params=params, auth=('admin', 'admin'))
            response.raise_for_status()

            dags_data = response.json()
            dags = dags_data.get('dags', [])

            if total_entries is None:
                total_entries = dags_data.get('total_entries', len(dags))

            if not dags:
                break

            for d in dags:
                dag_id = d.get('dag_id')
                is_paused_status = d.get('is_paused')
                all_dag_ids[dag_id] = is_paused_status

            offset += limit

    except requests.exceptions.RequestException as e:
        LOG.error(f"❌ Error fetching active DAGs from {base_url}: {e}")
        return {}

    LOG.info(f"✅ Found {len(all_dag_ids)} active DAGs.")
    return all_dag_ids


def send_slack_failure_check_notification(slack_conn_id: str, dag_id: str, run_id: str, message: str,
                                          full_log_content: str, is_latest: bool, days_to_check: int,
                                          is_paused: bool = False):
    """
    Sends a detailed, separate notification for one unresolved failure to Slack using SlackWebhookHook.
    The message header now USES the type determined by the core logic (New, Recurring, Resolved).
    """
    if not slack_conn_id:
        LOG.warning("⚠️ Slack Connection ID is not configured. Skipping Slack notification.")
        return

    dag_status = "ACTIVE DAG" if not is_paused else "PAUSED DAG"
    dag_status_emoji = "🟢" if not is_paused else "⏸️"

    type_match = re.search(r'\*([^*]+)\*\s*\n\*DAG:', message)

    date_match = re.search(r'\(at\s+(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s+UTC\)', message)

    core_failure_type = type_match.group(1).strip() if type_match else "UNKNOWN FAILURE TYPE"
    fail_time_str = date_match.group(1) if date_match else "UNKNOWN TIME"

    days_text = "day" if days_to_check == 1 else "days"
    analysis_window_text = f" (Run history analyzed over the last {days_to_check} {days_text})."

    if "RESOLVED" in core_failure_type.upper():
        header_emoji = "✅"
        failure_type_tag = "RESOLVED FAILURE"
        classification_rule = (
            f"Resolved - Current run: *SUCCESS* (automated or marked by developer). Previous run: *FAILED*. {analysis_window_text}"
        )
    elif "RECURRING" in core_failure_type.upper():
        header_emoji = "🟠"
        failure_type_tag = "RECURRING FAILURE"
        classification_rule = (
            f"Recurring - Current run: *FAILED*. Previous run: *FAILED*. {analysis_window_text}"
        )
    elif "NEW" in core_failure_type.upper():
        header_emoji = "🚨"
        failure_type_tag = "NEW FAILURE"
        classification_rule = (
            f"New - Current run: *FAILED*. Previous run: *SUCCESS* or *MISSING*. {analysis_window_text}"
        )

    else:
        header_emoji = "❓"
        failure_type_tag = core_failure_type
        classification_rule = f"Unknown: Status classification not defined.{analysis_window_text}"

    header_text = f"{header_emoji} {failure_type_tag} [{dag_status_emoji} {dag_status}]: {dag_id}"

    log_snippet = full_log_content[:MAX_SLACK_LOG_LENGTH] + (
        "\n... [LOG TRUNCATED] ..." if len(full_log_content) > MAX_SLACK_LOG_LENGTH else "")

    fallback_text = f"QA Failure Report: {failure_type_tag} {dag_id}"

    message_lines = message.strip().split('\n')

    indices_to_keep = [i for i, _ in enumerate(message.strip().split('\n')) if i not in [0, 1, 3]]

    cleaned_message = '\n'.join([message_lines[i] for i in indices_to_keep])

    blocks = [
        {"type": "divider"},
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": header_text,
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"ℹ️ *Status:* `{failure_type_tag}`\n*Rule:* {classification_rule}"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": cleaned_message
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Error Details (First 1500 characters):*"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"```\n{log_snippet}\n```"
            }
        },
        {"type": "divider"}
    ]

    try:
        hook = SlackWebhookHook(slack_webhook_conn_id=slack_conn_id)
        hook.send(text=fallback_text, blocks=blocks)
        LOG.info(f"Slack notification sent successfully for DAG: {dag_id} using connection {slack_conn_id}.")
    except Exception as e:
        LOG.error(f"Failed to send Slack notification for DAG {dag_id} using hook: {e}")


def get_run_datetime(run_data: Dict[str, Any], field: str = 'end_date') -> Optional[datetime]:
    """
    Extracts a timezone-aware datetime object (in UTC) from an Airflow DAG run object.
    """
    try:
        time_str = run_data.get(field)
        if not time_str:
            if field != 'execution_date':
                # Fallback to execution_date if end_date is missing
                return get_run_datetime(run_data, 'execution_date')
            return None

        dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        return dt.astimezone(timezone.utc)
    except Exception as e:
        LOG.error(f"❌ Error parsing time from field '{field}' for run '{run_data.get('dag_run_id')}': {e}")
        return None


def filter_emr_errors(log_content: str) -> str:
    """
    Filters EMR log content to show only ERROR Client: Application diagnostics messages.
    """
    pattern = re.compile(
        r"^\d{2}/\d{2}/\d{2}\s\d{2}:\d{2}:\d{2}\sERROR\sClient: Application diagnostics message: User class threw exception:.*?"
        r"(?=\n\d{2}/\d{2}/\d{2}\s\d{2}:\d{2}:\d{2}|$)",
        re.MULTILINE | re.DOTALL
    )
    matches = pattern.findall(log_content)
    if not matches:
        return "No specific 'ERROR Client: Application diagnostics message: User class threw exception:' blocks found."

    # Concatenating and returning the log
    full_filtered_log = "\n".join(matches)
    return full_filtered_log


def _get_api_headers() -> Dict[str, str]:
    """Provides headers for Airflow API requests."""
    return {"Content-Type": "application/json"}


def _clear_s3_path(s3_path: str, session: boto3.Session, **kwargs):
    LOG = logging.getLogger(__name__)
    LOG.setLevel(logging.DEBUG)

    if not s3_path.startswith("s3://"):
        LOG.warning(f"Invalid S3 path to clean up: {s3_path}")
        return

    import awswrangler as wr

    try:
        LOG.debug(f"CLEANUP: Removing the contents of the S3 prefix:{s3_path}")
        wr.s3.delete_objects(
            path=s3_path,
            boto3_session=session,
            use_threads=True
        )
        LOG.debug(f"CLEANUP: S3 contents of prefix {s3_path} successfully deleted.")
    except Exception as e:
        LOG.error(f"Critical error while trying to clear S3 path {s3_path}: {e}")
        pass


def execute_and_read_athena_query(boto3_session: boto3.Session, query_execution_id, **kwargs) -> pd.DataFrame:
    LOG = logging.getLogger(__name__)
    LOG.setLevel(logging.DEBUG)

    import awswrangler as wr

    LOG.debug(f"The query is being executed: {query_execution_id}")
    try:
        df_results = wr.athena.get_query_results(
            query_execution_id=query_execution_id,
            boto3_session=boto3_session
        )

        LOG.debug("The data has been successfully loaded into the DataFrame.")
        LOG.debug(df_results.head())

        return df_results

    except Exception as e:
        print(f"Error loading query results: {e}")


def datetime_serializer(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return None


def choose_branch_based_on_env(**kwargs):
    from cogent.utils.variables import AWS_ENV

    if AWS_ENV in TARGET_ENVS:
        return 'create_database_if_not_exists'
    else:
        LOG.warning(f"ENV '{AWS_ENV}' value is not allowed. Skip to the end.")
        return 'end_dag'


def determine_import_errors_flow(**kwargs):
    params = kwargs.get('params', {})

    should_run = params.get('run_import_errors', True)

    if should_run:
        return 'qa_check_airflow_import_errors'
    else:
        return 'import_errors_skip_signal'


def _get_error_details_for_run(
        dag_id: str, run_id: str, base_url: str, headers: Dict[str, str],
        emr_log_filtering_enabled: bool
) -> Optional[Dict[str, Any]]:
    task_instances_url = f"{base_url}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances"

    try:
        task_instances_resp = requests.get(task_instances_url, headers=headers)
        task_instances_resp.raise_for_status()
        task_instances = task_instances_resp.json().get('task_instances', [])
    except requests.exceptions.RequestException as e:
        LOG.error(f"❌ API error for task instances in run {run_id}: {e}")
        return None

    error_logs = []

    tasks_to_check = [task for task in task_instances if task['state'] == 'failed']
    if not tasks_to_check:
        tasks_to_check = [
            task for task in task_instances
            if task['state'] in ['skipped', 'upstream_failed', 'up_for_retry']
        ]

    for task in tasks_to_check:
        task_id = task['task_id']
        log_url = f"{base_url}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/1"

        try:
            log_resp = requests.get(log_url, headers=headers)
            log_resp.raise_for_status()
            log_content = log_resp.text
        except requests.exceptions.RequestException as e:
            log_content = f"Failed to retrieve log from Airflow API: {e}"

        log_report_content = ""
        emr_log_parser = EmrLogParser(log_content)
        emr_params = emr_log_parser.get_params()

        if emr_params:
            S3_LOG_BUCKET, CLUSTER_ID, STEP_ID = emr_params
            s3_key_path = f"emr_logs/{CLUSTER_ID}/steps/{STEP_ID}/{LOG_TYPE}.gz"
            s3_uri = f"s3://{S3_LOG_BUCKET}/{s3_key_path}"

            emr_header_prefix = f"| Cluster: {CLUSTER_ID} | Step: {STEP_ID}"

            emr_log_details = get_emr_log_content_from_s3(
                S3_LOG_BUCKET, CLUSTER_ID, STEP_ID, LOG_TYPE
            )

            if emr_log_details.startswith("❌"):
                # EMR fetch failed. Fallback to Airflow logs.
                error_lines = [
                    line for line in log_content.split('\n')
                    if any(keyword in line.lower() for keyword in AIRFLOW_ERROR_KEYWORDS)
                ]
                log_report_content = (
                        f"--- Airflow Error Fallback {emr_header_prefix} ---\n" +
                        f"{emr_log_details}\n\n" + "\n".join(error_lines) +
                        f"\n\n🔗 Full S3 log link (Access failed): {s3_uri}"
                )
            else:
                # EMR fetch successful: Apply filtering
                if emr_log_filtering_enabled:
                    filtered_emr_log = filter_emr_errors(emr_log_details)
                    log_display_type = "FILTERED LOG (Application Exceptions)"
                else:
                    filtered_emr_log = emr_log_details
                    log_display_type = "FULL EMR LOG"

                log_report_content = (
                    f"--- EMR {log_display_type} {emr_header_prefix} ---\n"
                    f"{filtered_emr_log}\n"
                    f"🔗 Full S3 log link: {s3_uri}"
                )

        else:
            # EMR log link NOT found: Fallback to improved Airflow log filtering
            error_lines = []
            for line in log_content.split('\n'):
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in AIRFLOW_ERROR_KEYWORDS):
                    error_lines.append(line)

            if not error_lines:
                # Fallback to the last Traceback error block
                traceback_match = re.search(r"Traceback \(most recent call last\):.*", log_content,
                                            re.DOTALL | re.IGNORECASE)
                if traceback_match:
                    traceback_lines = traceback_match.group(0).split('\n')
                    traceback_lines = [line for line in traceback_lines if
                                       not line.strip().startswith(('ip-', '***', '▲▲▲', '▼▼▼', '[202'))]
                    error_lines.extend(traceback_lines)

            if not error_lines:
                error_lines = [line for line in log_content.split('\n') if line.strip().startswith('ERROR -')]

            full_log = "\n".join(error_lines)
            log_report_content = full_log

        if log_report_content.strip():
            error_logs.append({
                "task_id": task_id,
                "state": task['state'],
                "logs": log_report_content
            })

    if error_logs:
        full_dag_log = "\n".join([f"[{e['task_id']} - {e['state']}]\n{e['logs']}\n" for e in error_logs])

        return {
            "error_logs": error_logs,
            "full_dag_log": full_dag_log,
        }

    return None


def get_failed_runs_for_dags_core(
        base_url: str, dag_ids: List[str], auth_header: Optional[str], days_to_check: int,
        emr_log_filtering_enabled: bool
) -> Dict[str, List[Dict[str, Any]]]:
    latest_runs_report = {}

    headers = {"Content-Type": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header

    for dag_id in dag_ids:
        LOG.info(f"Checking DAG: {dag_id}...")

        dag_runs_url = f"{base_url}/api/v1/dags/{dag_id}/dagRuns"

        params_latest = {
            "order_by": "-execution_date",
            "limit": 2
        }

        try:
            response_latest = requests.get(dag_runs_url, headers=headers, params=params_latest)
            response_latest.raise_for_status()
            all_runs = response_latest.json().get('dag_runs', [])
        except requests.exceptions.RequestException as e:
            LOG.error(f"❌ API error fetching runs for DAG {dag_id}: {e}")
            continue

        if not all_runs:
            # LOG.info(f"    - No runs found for DAG {dag_id}. Skipping.")
            continue

        current_run = all_runs[0]
        prev_run = all_runs[1] if len(all_runs) > 1 else None

        current_state = current_run.get('state')
        prev_run_state = prev_run.get('state') if prev_run else "missing"

        report_status = None
        failure_type = None
        run_to_report = None

        if current_state == 'failed':
            if prev_run_state == 'failed':
                report_status = "UNRESOLVED"
                failure_type = "Recurring Failure"
            else:  # (prev_run_state == 'success' or 'missing')
                report_status = "UNRESOLVED"
                failure_type = "New Failure"

            run_to_report = current_run

            LOG.info(
                f"   Classified as UNRESOLVED, Type: {failure_type} included in Slack report for run: {run_to_report.get('dag_run_id')}")

        elif current_state == 'success' and prev_run_state == 'failed':

            report_status = "RESOLVED"
            failure_type = "Resolved Failure"
            run_to_report = prev_run

            LOG.info(
                f"   Classified as RESOLVED, Type: {failure_type} included in Slack report for run: {run_to_report.get('dag_run_id')}")

        else:
            continue

        if run_to_report:
            run_id_to_report = run_to_report.get('dag_run_id')
            header_emoji = "✅" if report_status == "RESOLVED" else "❌"

            current_error_details = _get_error_details_for_run(
                dag_id, run_id_to_report, base_url, headers, emr_log_filtering_enabled
            )

            if not current_error_details or not current_error_details.get('error_logs'):
                LOG.warning(f"   Could not retrieve error details for run {run_id_to_report}. Skipping.")
                continue

            latest_runs_report[dag_id] = [{
                "run_id": run_id_to_report,
                "status_type": report_status,
                "failure_type": failure_type,
                "errors": current_error_details['error_logs'],
                "run_data": run_to_report,
                "full_log_content": current_error_details['full_dag_log'],
                "is_latest_for_dag": True,
                "header_emoji": header_emoji
            }]

    return latest_runs_report


def execute_pylint_analysis(
        nr_config_path: str = None,
        airflow_api_url: str = None,
        dag_id: str = None,
        is_debug_mode: bool = False,
        dag_ids_from_config: 'Optional[List[str]]' = None,
        **kwargs
) -> dict:
    """
    Runs Pylint on all DAG files and returns all structured Pylint issues.

    NEW RELIC INTEGRATION:
    - Initializes NR agent using a provided .ini config path or env var.
    - Registers the app as 'Airflow-Pylint-Analysis'.
    - Records custom parameters (execution_mode, airflow_dag).
    - Captures execution errors via notice_error.
    - Sends a summary event 'PylintExecutionSummary' with scores.
    - Ensures data flush via shutdown_agent in the finally block.
    """
    import os
    import sys

    path_override = kwargs.get('python_path_override')

    if path_override:
        os.environ['PYTHONPATH'] = path_override

        paths = path_override.split(':')
        for path in paths:
            if path and path not in sys.path:
                sys.path.insert(0, path)

    kwargs.pop('python_path_override', None)

    print(f"[INFO] PYTHONPATH: {os.environ['PYTHONPATH']}")

    from jobs.airflow_pylint_qa.nr_decorators import with_nr_monitoring

    @with_nr_monitoring(task_name="Pylint-Active-Analysis")
    def run_analysis(nr_config_path=None, dag_id=None, api_url=None, is_debug=False):
        import newrelic.agent

        try:

            import json
            from typing import List, Dict
            from datetime import datetime, timezone
            from dataclasses import asdict

            from jobs.airflow_pylint_qa.pylint_checker import PylintQualityChecker
            from jobs.airflow_pylint_qa.regression import PylintIssue

            current_api_url = api_url or "http://localhost:8080/api/v1"

            # dags_to_ignore should include this DAG itself and common examples
            dags_to_ignore = [dag_id, 'airflow_pylint_qa', 'execute_spark_query', 'example_dag', 'airflow_db_cleanup']
            checker = PylintQualityChecker(
                dags_to_ignore=dags_to_ignore,
                api_url=current_api_url,
                is_debug_mode=is_debug
            )
            all_issues: List[PylintIssue] = []
            current_time = datetime.now(timezone.utc).replace(microsecond=0)
            scores_by_dag: Dict[str, float] = {}  # For tracking the score of each DAG

            # Determine which DAGs to analyze: from config parameter or all active DAGs via API
            print(f"[INFO] Input DAG list: {dag_ids_from_config}")
            dag_ids_list: List[str] = []

            if dag_ids_from_config is not None:
                if isinstance(dag_ids_from_config, str):
                    try:
                        import json
                        temp_list = json.loads(dag_ids_from_config.replace("'", "\""))
                        if isinstance(temp_list, list) and all(isinstance(i, str) for i in temp_list):
                            dag_ids_list = temp_list
                        else:
                            dag_ids_list = [dag_ids_from_config]
                    except (json.JSONDecodeError, TypeError):
                        dag_ids_list = [dag_ids_from_config]
                elif isinstance(dag_ids_from_config, list):
                    dag_ids_list = dag_ids_from_config

            if isinstance(dag_ids_list, list) and dag_ids_list:
                print(f"[INFO] Analyzing {len(dag_ids_list)} DAG(s) from input configuration.")
                dags_to_analyze = dag_ids_list
            else:
                print("[INFO] Input list is empty or invalid. Switching to analyzing ALL active DAGs.")
                dags_to_analyze = checker.get_all_dags()

            total_score = 0.0
            analyzed_dags_count = 0
            total_dags = len(dags_to_analyze)

            for current_dag in dags_to_analyze:
                print(f"[INFO] Analyzing DAG file: {current_dag}.py ({analyzed_dags_count + 1} of {total_dags})")

                try:
                    # Pylint needs the file path relative to its execution context, typically dags/{dag_id}.py
                    dag_file = os.path.join(checker.dag_folder, f"{current_dag}.py")

                    if not os.path.exists(dag_file):
                        print(f"[WARNING] DAG file not found locally: {dag_file}. Skipping.")
                        continue

                    # Execute Pylint
                    output, score = checker.run_pylint(dag_file)

                    if score is not None:
                        total_score += score
                        scores_by_dag[current_dag] = score

                    analyzed_dags_count += 1

                    # Parse Pylint output into structured issues
                    issues = checker.parse_pylint_output(output, current_dag, current_time, score)
                    all_issues.extend(issues)

                except Exception as e:
                    print(f"[ERROR] Failed to run Pylint for DAG {current_dag}: {e}")

            # Calculate final score (default to 10.0 if no DAGs were analyzed to prevent division by zero)
            final_score = total_score / analyzed_dags_count if analyzed_dags_count > 0 else 10.0

            # Collect results for XCom
            serializable_issues = [asdict(i) for i in all_issues]

            try:
                newrelic.agent.record_custom_event('PylintExecutionSummary', {
                    'dag_id': dag_id,
                    'final_score': final_score,
                    'analyzed_count': analyzed_dags_count,
                    'total_issues': len(all_issues)
                })
            except Exception as nr_e:
                print(f"[NR ERROR] Failed to record event: {nr_e}")

            return {
                'all_issues': serializable_issues,
                'current_pylint_score': final_score,
                'scores_by_dag': scores_by_dag,  # Scores for detailed report
                'analyzed_count': analyzed_dags_count,
            }
        except Exception as e:
            print(f"[ERROR] Logic failed: {e}")

            newrelic.agent.notice_error()

            return {
                "status": "api_failed",
                "all_issues": [],
                "scores": {}
            }

    return run_analysis(
        nr_config_path=nr_config_path,
        dag_id=dag_id,
        api_url=airflow_api_url,
        is_debug=is_debug_mode
    )


def check_airflow_import_errors_and_prepare_report(
        api_url: str,
        env_name: str,
        is_debug_mode: bool,
        python_path_override: str,
        **kwargs
) -> Optional[Dict[str, Any]]:
    if python_path_override:
        os.environ['PYTHONPATH'] = python_path_override

    paths = python_path_override.split(':')
    for path in paths:
        if path and path not in sys.path:
            sys.path.insert(0, path)

    kwargs.pop('python_path_override', None)

    from jobs.airflow_pylint_qa.pylint_checker import PylintQualityChecker

    checker = PylintQualityChecker(
        api_url=api_url,
        env_name=env_name,
        is_debug_mode=is_debug_mode)
    slack_payload = checker.check_import_errors()

    return slack_payload


@task(task_id='process_slack_payload')
def process_slack_payload(api_url: str, **kwargs) -> Dict[str, Any]:
    AIRFLOW_URL_IMPORT_ERRORS = f"{api_url}/api/v1/importErrors?limit=100"
    AIRFLOW_LINK = f"<{AIRFLOW_URL_IMPORT_ERRORS}|Airflow UI>"
    SUCCESS_SUBSTRING = "SUCCESSFUL. Broken DAGs: 0."

    NICELY_FORMATTED_SUCCESS = (
        "\n🎉 *Airflow QA Import Errors Report: SUCCESS* 🎉\n\n"
        f":heavy_check_mark: *Status check for {AIRFLOW_LINK}:* SUCCESSFUL.\n"
        ":heavy_check_mark: Broken DAGs: 0.\n\n"
        "---\n"
        ":trophy: *All DAGs loaded successfully.* Import quality meets requirements."
    )

    ti = kwargs.get('ti')
    report_data = ti.xcom_pull(task_ids='qa_check_airflow_import_errors')

    if report_data is None:
        return {"slack_messages": [], "report_granularity": "single"}

    message_text = report_data.get('message', "Error: Unknown message.")
    granularity = report_data.get('granularity', 'single')

    if SUCCESS_SUBSTRING in message_text:
        message_text = NICELY_FORMATTED_SUCCESS

    slack_messages: List[str] = [message_text]

    return {
        "slack_messages": slack_messages,
        "report_granularity": granularity
    }


def check_for_regression(
        pylint_results_task_id: str,
        historical_data_task_id: str,
        **kwargs
) -> Dict[str, Any]:
    """
    Compares current PyLint scores with historical (N-1) scores to detect regression.
    A regression is detected if the current DAG score is lower than the historical score (within tolerance).
    """

    LOG = logging.getLogger(__name__)
    LOG.setLevel(logging.DEBUG)

    ti = kwargs['ti']

    current_results: Dict[str, Any] = ti.xcom_pull(task_ids=pylint_results_task_id, key='return_value')
    historical_data: List[Dict[str, Any]] = ti.xcom_pull(task_ids=historical_data_task_id, key='return_value')

    LOG.debug("=== INPUT DATA check_for_regression ===")
    LOG.debug(
        "current_results: \n%s",
        json.dumps(current_results, indent=2, default=datetime_serializer)
    )
    LOG.debug(
        "historical_data: \n%s",
        json.dumps(historical_data, indent=2, default=datetime_serializer)
    )
    LOG.debug("=======================================")

    if historical_data is None:
        LOG.warning("Historical data None. Using an empty list for comparison.")
        historical_data = []

    current_scores: Dict[str, float] = current_results['scores_by_dag']
    regression_dags: Dict[str, Dict[str, float]] = {}

    historical_scores: Dict[str, float] = {}

    for data in historical_data:
        try:
            score = float(data.get('overall_score'))
            historical_scores[data['dag_id']] = score
        except (ValueError, TypeError):
            LOG.warning(f"The historical score for DAG {data.get('dag_id')} is not a number. Omitted.")
            continue

    for dag_id, current_score in current_scores.items():

        prev_score = historical_scores.get(dag_id, None)

        if prev_score is None:
            comparison_score = 10.0

            LOG.debug(
                f"DAG '{dag_id}': Regression check skipped (No N-1 baseline). "
                f"Comparing current score {current_score:.2f} against fallback {comparison_score:.2f}."
            )

            if current_score < comparison_score - 0.1:
                regression_dags[dag_id] = {
                    'current_score': current_score,
                    'previous_score': comparison_score,  # 10.0
                    'difference': current_score - comparison_score
                }

            continue

        LOG.info(
            f"DAG '{dag_id}': Comparing scores: "
            f"Current={current_score:.2f}, Previous (N-1)={prev_score:.2f}."
        )

        if current_score < prev_score - 0.1:
            regression_dags[dag_id] = {
                'current_score': current_score,
                'previous_score': prev_score,
                'difference': current_score - prev_score
            }

    current_results['regression_dags'] = regression_dags
    current_results['regression_found'] = bool(regression_dags)

    LOG.debug(f"Regression detected in {len(regression_dags)} DAGs.")

    return current_results


def prepare_regression_slack_messages(
        airflow_api_url: str,
        regression_report_task_id: str,
        **kwargs
) -> List[Any]:
    """
    Generates Slack messages for REGRESSION ALERTS ONLY.

    If regression is found, it sends a detailed alert.
    If no regression is found, it sends a simple confirmation message.
    """
    ti = kwargs['ti']
    GRANULARITY_TASK_ID = 'extract_report_granularity'

    report_granularity: str = ti.xcom_pull(
        task_ids=GRANULARITY_TASK_ID,
        key='return_value'
    )

    if not report_granularity:
        report_granularity = "single"

    pylint_results: dict = ti.xcom_pull(
        task_ids=regression_report_task_id,
        key='return_value'
    )

    LOG = logging.getLogger(__name__)
    LOG.setLevel(logging.DEBUG)

    scores_by_dag = pylint_results['scores_by_dag']
    all_issues = pylint_results['all_issues']
    regression_found = pylint_results.get('regression_found', False)
    regression_dags = pylint_results.get('regression_dags', {})

    slack_messages: List[Any] = []

    # Prepare base URL
    base_url = airflow_api_url.replace('/api/v1/', '').replace('/api/v1', '').rstrip('/')

    LOG.debug(f"Pylint results loaded. Regression found: {regression_found}, Report Granularity: {report_granularity}")

    # --- KEY CONTROL: EXIT CASE WHEN NO REGRESSION ---
    if not regression_found:
        LOG.debug("No regression detected. Generating confirmation message.")

        confirmation_text = ":sparkles: Compared to the previous run, *no changes in code quality were detected*. Quality remains stable."

        if report_granularity == 'single':
            # Block Kit structure for 'single' mode
            blocks: List[Dict[str, Any]] = [
                {"type": "divider"},
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "✅ Pylint QA: Regression Report"
                    }
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": confirmation_text}
                },
                {"type": "divider"},
            ]
            slack_messages.append(blocks)
        else:  # 'multiple' mode uses markdown text
            message = f"**Pylint QA: Regression Report**: {confirmation_text}"
            slack_messages.append(message)

        return slack_messages

    # --- LOGIC FOR DETECTED REGRESSION ---

    LOG.debug(f"Regression detected in {len(regression_dags)} DAGs. Generating alert messages.")

    # --- 1. SINGLE (Detailed Composite Report using Block Kit) ---
    if report_granularity == 'single':

        reg_count = len(regression_dags)
        header_icon = ":fire:"

        # 1a. Format the detailed per-DAG section content (ONLY for regressed DAGs)
        details_list = []
        for dag_id, reg_data in regression_dags.items():
            score = scores_by_dag.get(dag_id, 0.0)
            prev_score = reg_data['previous_score']
            diff = reg_data['difference']

            dag_url = f"{base_url}/dags/{dag_id}/grid"

            # Focus on regression
            # Using double backticks to escape them in f-strings for mrkdwn output
            display_text = f"`{dag_id}`: *{score:.1f} / 10.0* (Degradation by `{diff:.1f}`. Was: `{prev_score:.1f}`)"
            linked_item = f"<{dag_url}|{display_text}>"

            details_list.append(f"• {linked_item}")

        details_content = "\n".join(details_list)

        # 1b. Construct the ALERT Block Kit payload
        summary_text = (
            f"Pylint Score regression detected in *{reg_count} DAG(s)* compared to the previous successful run. "
            f"Please review the details and apply fixes.\n"
        )

        blocks: List[Dict[str, Any]] = [
            {"type": "divider"},
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 Pylint QA: REGRESSION ALERT"
                }
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": summary_text}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*📉 Details for {reg_count} Regressed DAGs:*\n{details_content}"}
            },
            {"type": "divider"}
        ]

        slack_messages.append(blocks)

    # --- 2. MULTIPLE (One message per DAG for REGRESSED DAGS ONLY) ---
    elif report_granularity == 'multiple':

        # 2a. Group violations by DAG ID
        issues_by_dag: Dict[str, List[Dict[str, Any]]] = {}
        for issue in all_issues:
            dag_id = issue['dag_id']
            if dag_id in regression_dags:  # Filter ONLY REGRESSED
                if dag_id not in issues_by_dag:
                    issues_by_dag[dag_id] = []
                issues_by_dag[dag_id].append(issue)

        # 2b. Create a message for each REGRESSED DAG
        for dag_id, issues in issues_by_dag.items():
            dag_score = scores_by_dag.get(dag_id, 10.0)
            reg_data = regression_dags[dag_id]  # Guaranteed to exist, thanks to the filter
            prev_score = reg_data['previous_score']
            diff = reg_data['difference']

            status_icon = ":fire:"
            quality_check = f"Quality: *REGRESSION ({prev_score:.1f} -> {dag_score:.1f} / {diff:.1f})*."

            dag_url = f"{base_url}/dags/{dag_id}/grid"
            linked_dag_id = f"<{dag_url}|`{dag_id}`>"

            issue_list = []
            for issue in issues:
                # Issue details remain in standard format
                issue_list.append(
                    f"  • *{issue['symbol_code']} ({issue['issue_type']})* | Line *{issue['line_number']}*: {issue['message']}"
                )

            # Limit the list length to avoid overwhelming Slack
            issue_display = '\n'.join(issue_list[:10])
            if len(issue_list) > 10:
                issue_display += f"\n  _...and {len(issue_list) - 10} more violations._"

            message = (
                f"\n\n{status_icon} *REGRESSION REPORT for DAG: {linked_dag_id}:*\n"
                f"*Score:* `{dag_score:.1f}/10.0` | {quality_check} \n\n"
                f"*Violations Found ({len(issues)}):*\n"
                f"{issue_display}"
            )
            slack_messages.append(message)

    LOG.debug(f"Regression Alert messages prepared. Total messages: {len(slack_messages)}")
    return slack_messages


@task(task_id='send_regression_slack_report')
def send_regression_slack_report(
        slack_messages: List[Any],
        report_granularity: str,
        **kwargs):
    """
    Sends the list of prepared messages to Slack. Uses Block Kit for 'single' reports
    and simple text for 'multiple' reports.
    """

    LOG = logging.getLogger(__name__)
    LOG.setLevel(logging.DEBUG)

    try:
        slack_hook = SlackWebhookHook(slack_webhook_conn_id=SLACK_WEBHOOK_CONN_ID)

        # Handle 'single' mode (Block Kit list) vs 'multiple' mode (text list)
        if report_granularity == 'single' and slack_messages and isinstance(slack_messages[0], list):
            # Single mode: slack_messages is a list containing ONE item: a list of blocks.
            blocks_payload = slack_messages[0]
            LOG.debug("Sending Slack message using Block Kit (Single Report)...")

            # Send blocks and a fallback text message (required by Slack API)
            fallback_text = "Pylint QA Report (See details in Block Kit layout)"
            slack_hook.send(text=fallback_text, blocks=blocks_payload)

            LOG.info("[INFO] All Slack messages sent successfully.")

        else:  # Handles 'multiple' mode and edge cases
            # Multiple mode: slack_messages is a list of plain text strings.
            for i, message in enumerate(slack_messages):
                LOG.debug(f"Sending Slack message {i + 1} of {len(slack_messages)} (Text Report)...")
                # The 'message' is already formatted as Slack markdown text
                slack_hook.send(text=message)
            LOG.info("[INFO] All Slack messages sent successfully.")

    except Exception as e:
        LOG.error(
            f"CRITICAL ERROR: Failed to send Slack report. Check if '{SLACK_WEBHOOK_CONN_ID}' connection is set up correctly. Error: {e}",
            exc_info=True)


def create_iceberg_table_ddl(**kwargs) -> str:
    """
    Creates the Iceberg DDL query using the CTAS Workaround.
    Defines the schema including overall_score.
    """
    iceberg_table_ddl = f"""
        CREATE TABLE IF NOT EXISTS {DATABASE}.{TABLE}
        WITH (
            table_type = 'ICEBERG',
            is_external = false,
            location = '{ICEBERG_TABLE_LOCATION}/v1'
        ) AS
        -- Defining the schema via a dummy SELECT (CTAS)
        SELECT 
            CAST(NULL AS VARCHAR) AS dag_id,
            CAST(NULL AS TIMESTAMP) AS check_timestamp,
            CAST(NULL AS DOUBLE) AS overall_score, 
            CAST(NULL AS VARCHAR) AS issue_type,
            CAST(NULL AS VARCHAR) AS symbol,
            CAST(NULL AS VARCHAR) AS symbol_code,
            CAST(NULL AS VARCHAR) AS message,
            CAST(NULL AS INTEGER) AS line_number,
            CAST(NULL AS VARCHAR) AS result_of_last_run,
            CAST(NULL AS TIMESTAMP) AS time_of_last_run
        WHERE 1 = 0
    """
    return iceberg_table_ddl


def extract_report_granularity(**kwargs) -> str:
    """Extracts the report_granularity parameter from DAG run configuration."""
    # Get the parameter from the DAG context (defaulting to 'single')
    import logging
    LOG = logging.getLogger(__name__)
    granularity = kwargs.get("params", {}).get("report_granularity", "single")
    LOG.debug(f"Slack report granularity set to: {granularity}")
    return granularity


def prepare_slack_messages(
        airflow_api_url: str,
        pylint_results_task_id: str,
        **kwargs) -> List[Any]:
    """
    Generates Slack messages (in Block Kit structure for 'single', or markdown for 'multiple').
    The 'single' mode generates one message using Slack blocks, now with hyperlinks.
    """
    ti = kwargs['ti']
    GRANULARITY_TASK_ID = 'extract_report_granularity'

    report_granularity: str = ti.xcom_pull(
        task_ids=GRANULARITY_TASK_ID,
        key='return_value'
    )

    if not report_granularity:
        report_granularity = "single"

    pylint_results: dict = ti.xcom_pull(task_ids=pylint_results_task_id, key='return_value')

    global_score = pylint_results['current_pylint_score']
    all_issues = pylint_results['all_issues']
    scores_by_dag = pylint_results['scores_by_dag']
    analyzed_count = pylint_results['analyzed_count']

    slack_messages: List[Any] = []

    base_url = airflow_api_url

    if base_url.endswith('/api/v1/'):
        base_url = base_url[:-len('/api/v1/')]
    elif base_url.endswith('/api/v1'):
        base_url = base_url[:-len('/api/v1')]

    base_url = base_url.rstrip('/')

    PYLINT_URL = "http://pylint.readthedocs.io/en/stable"
    PYLINT_LINK = f"<{PYLINT_URL}|_(Calculated using the internal Pylint algorithm)_>"

    # --- 1. SINGLE (Detailed Composite Report using Block Kit) ---
    if report_granularity == 'single':

        # 1a. Determine overall status and summary
        if global_score >= PYLINT_QUALITY_TARGET:
            quality_status = f":heavy_check_mark: *Quality meets* the target threshold (`{PYLINT_QUALITY_TARGET}/10.0`)."
            regression_info = ":sparkles: *No Regression Detected.*"
            header_icon = ":white_check_mark:"
        else:
            quality_status = f":rotating_light: *Quality is below* the target threshold (`{PYLINT_QUALITY_TARGET}/10.0`)."
            regression_info = ":fire: *Potential Regression* (average score is low)."
            header_icon = ":warning:"

        # Aggregating total errors/warnings
        total_errors = sum(1 for i in all_issues if i['issue_type'] in ['E', 'F'])
        total_warnings = sum(1 for i in all_issues if i['issue_type'] == 'W')

        # 1b. Format the detailed per-DAG section content
        details_list = []
        sorted_scores = sorted(scores_by_dag.items(), key=lambda item: item[1])

        for dag_id, score in sorted_scores:
            dag_url = f"{base_url}/dags/{dag_id}/grid"
            display_text = f"`{dag_id}`: *{score:.1f} / 10.0*"
            linked_item = f"<{dag_url}|{display_text}>"
            details_list.append(f"• {linked_item}")

        details_content = "\n".join(details_list)

        max_chars = 2500
        if len(details_content) > max_chars:
            details_content = details_content[:max_chars]
            details_content += f"\n\n... and {len(details_list)} total DAGs (truncated for Slack size limit)."

        # 1c. Construct the Block Kit payload
        summary_text = (
            f"Analysis completed for *{analyzed_count} DAGs*. \n"
            f"*:trophy: Overall Pylint Score (Average):* *`{global_score:.1f}/10.0`*. {PYLINT_LINK}\n"
            f"{quality_status}\n"
            f"*{regression_info}*\n"
        )

        blocks: List[Dict[str, Any]] = [
            {"type": "divider"},
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Pylint QA Report (Summary)"
                }
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": summary_text}
            },
            {"type": "divider"},
        ]

        if analyzed_count > 0 and details_content.strip():
            blocks.extend([
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "*📊 Detailed DAG Scores: (Lowest to Highest)*"}
                },
                # Detailed DAG Scores Content
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"... {len(details_list)} total DAGs (truncated for Slack size limit)."
                        # "text": details_content
                    }
                },
                {"type": "divider"},
            ])

        blocks.append({
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f":x: Critical Errors/Fatal: *{total_errors}*"},
                {"type": "mrkdwn", "text": f":warning: Warnings: *{total_warnings}*"}
            ]
        })

        # For 'single' report, we return one item which is the list of blocks
        slack_messages.append(blocks)

    # --- 2. MULTIPLE (One message per DAG with violations - Remains text-based) ---
    elif report_granularity == 'multiple':

        # Group violations by DAG ID
        issues_by_dag: Dict[str, List[Dict[str, Any]]] = {}
        for issue in all_issues:
            dag_id = issue['dag_id']
            # Initialize list for new DAG ID
            if dag_id not in issues_by_dag:
                issues_by_dag[dag_id] = []
            issues_by_dag[dag_id].append(issue)

        # Create a message for each DAG that has violations
        for dag_id, issues in issues_by_dag.items():
            dag_score = scores_by_dag.get(dag_id, 10.0)

            if dag_score >= PYLINT_QUALITY_TARGET:
                status_icon = ":heavy_check_mark:"
                quality_check = f"Quality: OK. (`{dag_score:.1f}/10.0`)"
            else:
                status_icon = ":rotating_light:"
                quality_check = f"Quality: *LOW.* (`{dag_score:.1f}/10.0`)"

            dag_url = f"{base_url}/dags/{dag_id}/grid"
            linked_dag_id = f"<{dag_url}|`{dag_id}`>"
            issue_list = []
            for issue in issues:
                issue_list.append(
                    f"  • *{issue['symbol_code']} ({issue['issue_type']})* | Line *{issue['line_number']}*: {issue['message']}"
                )

            # Limit the list length to avoid overwhelming Slack
            issue_display = '\n'.join(issue_list[:10])
            if len(issue_list) > 10:
                issue_display += f"\n  _...and {len(issue_list) - 10} more violations._"

            message = (
                f"\n{status_icon} *Pylint QA Report for DAG: {linked_dag_id}:*\n"
                f"*Score:* `{dag_score:.1f}/10.0` | {quality_check} {PYLINT_LINK} \n\n"
                f"*Found Violations ({len(issues)}):*\n"
                f"{issue_display}"
            )
            slack_messages.append(message)

        # If no violations were found but DAGs were analyzed, send a summary report
        if not slack_messages and analyzed_count > 0:
            slack_messages.append(
                f":sparkles: *Pylint QA Report* :sparkles:\n"
                f"Analysis of *{analyzed_count} DAGs* completed. \n"
                f"No violations (Errors, Warnings, Refactors, Conventions) were found that were not disabled by configuration. \n"
                f"Average Score: `{global_score:.1f}/10.0`."
            )

    LOG.debug(f"Messages prepared for Slack. Total messages: {len(slack_messages)}")
    return slack_messages


def send_slack_report(
        slack_messages_task_id: str,
        **kwargs):
    """
    Sends the list of prepared messages to Slack. Uses Block Kit for 'single' reports
    and simple text for 'multiple' reports.
    """

    ti = kwargs['ti']
    GRANULARITY_TASK_ID = 'extract_report_granularity'

    raw_xcom_data = ti.xcom_pull(task_ids=slack_messages_task_id, key='return_value')

    slack_messages: List[Any] = []
    report_granularity: str = "single"

    if isinstance(raw_xcom_data, dict) and 'slack_messages' in raw_xcom_data:
        slack_messages = raw_xcom_data.get("slack_messages", [])
        report_granularity = raw_xcom_data.get("report_granularity", "single")
        LOG.debug(f"XCom data unpacked from dictionary structure. Granularity: {report_granularity}")

    elif isinstance(raw_xcom_data, list):
        slack_messages = raw_xcom_data

        report_granularity = ti.xcom_pull(
            task_ids=GRANULARITY_TASK_ID,
            key='return_value'
        )
        if not report_granularity:
            report_granularity = "single"

    else:
        LOG.info(
            f"XCom data from {slack_messages_task_id} is empty or unexpected type ({type(raw_xcom_data)}). Skipping report.")
        return

    is_single_block_report = (
            report_granularity == 'single' and
            isinstance(slack_messages, list) and
            slack_messages and
            isinstance(slack_messages[0], list)
    )

    try:
        slack_hook = SlackWebhookHook(slack_webhook_conn_id=SLACK_WEBHOOK_CONN_ID)

        if is_single_block_report:
            blocks_payload = slack_messages[0]

            if blocks_payload and isinstance(blocks_payload, list) and blocks_payload and isinstance(blocks_payload[0],
                                                                                                     list):
                LOG.warning("Unpacking extra level of list nesting for Slack blocks.")
                blocks_payload = blocks_payload[0]

            if blocks_payload and blocks_payload[0].get('type') == 'header':
                header_text_dict = blocks_payload[0].get('text', {})
                if header_text_dict.get('type') == 'plain_text':
                    header_text_dict['text'] = header_text_dict.get('text', '').strip()

            if not blocks_payload:
                LOG.error("Attempted to send an empty Slack blocks array. Skipping.")
                return

            try:
                blocks_payload_json = json.dumps(blocks_payload, indent=2)
                LOG.info(f"Final Slack Blocks JSON for testing (check in Block Kit Builder):\n{blocks_payload_json}")
            except Exception as json_err:
                LOG.error(f"Failed to serialize Slack blocks for logging: {json_err}", exc_info=True)

            fallback_text = "Pylint QA Report (See details in Block Kit layout)"
            slack_hook.send(text=fallback_text, blocks=blocks_payload)
            LOG.debug("Slack message sent successfully via Block Kit.")

        else:
            if not slack_messages:
                LOG.info("No text-based Slack messages to send.")
                return

            for i, message in enumerate(slack_messages):
                LOG.debug(f"Sending Slack message {i + 1} of {len(slack_messages)} (Text Report)...")
                slack_hook.send(text=message)
            LOG.debug("All Slack messages sent successfully.")

    except Exception as e:
        LOG.error(
            f"CRITICAL ERROR: Failed to send Slack report. Check if '{SLACK_WEBHOOK_CONN_ID}' connection is set up correctly. Error: {e}",
            exc_info=True)


def get_active_dags_for_failure_check(**context) -> List[str]:
    # The logic fetches DAG list using Airflow API
    from cogent.utils.variables import AWS_ENV

    # base_url = f"http://edm-cogent-airflow.example{AWS_ENV}.int"
    base_url = get_airflow_base_url()
    user = 'admin'
    password = 'admin'
    auth_header = None
    if user and password:
        auth_string = f"{user}:{password}".encode("utf-8")
        auth_header = f"Basic {base64.b64encode(auth_string).decode('utf-8').strip()}"
    LOG.info(f"Attempting to fetch DAG list from: {base_url} for failure check.")

    return get_all_active_dags(base_url, auth_header=None)


def check_unresolved_failures_task_integrated(dags_list_task_id: str, **context) -> Dict[str, Any]:
    from cogent.utils.variables import AWS_ENV
    import base64
    from typing import List, Dict
    from datetime import datetime, timezone

    dag_run_params = context['params']
    days_to_check = dag_run_params['days_to_check']
    emr_log_filtering_enabled = True

    # base_url = f"http://edm-cogent-airflow.example{AWS_ENV}.int"
    base_url = get_airflow_base_url()
    slack_conn_id = SLACK_WEBHOOK_CONN_ID

    user = 'airflow'
    password = 'airflow'
    auth_header = None
    if user and password:
        auth_string = f"{user}:{password}".encode("utf-8")
        auth_header = f"Basic {base64.b64encode(auth_string).decode('utf-8').strip()}"

    ti = context['ti']
    # dag_ids: List[str] = ti.xcom_pull(task_ids=dags_list_task_id, key='return_value')
    dags_status: Dict[str, bool] = ti.xcom_pull(task_ids=dags_list_task_id, key='return_value')

    if not dags_status:
        LOG.warning(f"⚠️ XCom pull from {dags_list_task_id} returned no DAG IDs. Exiting.")
        return {}

    dag_ids: List[str] = list(dags_status.keys())

    full_runs_report = get_failed_runs_for_dags_core(
        base_url, dag_ids, auth_header, days_to_check, emr_log_filtering_enabled
    )

    if full_runs_report:

        all_failures_list = []
        for dag_id, runs in full_runs_report.items():
            all_failures_list.extend(runs)

        LOG.info(f"📊 Aggregated {len(all_failures_list)} total failed runs for reporting.")

        sorted_all_failures = sorted(
            all_failures_list,
            key=lambda run: get_run_datetime(run['run_data'], 'end_date') or datetime(1970, 1, 1,
                                                                                      tzinfo=timezone.utc),
            reverse=True
        )

        num_total_failures = len(sorted_all_failures)
        num_unresolved = sum(1 for r in sorted_all_failures if r['status_type'] == 'UNRESOLVED')
        num_resolved = num_total_failures - num_unresolved

        if sorted_all_failures:
            latest_run_data = sorted_all_failures[0]
            latest_dag_id = latest_run_data['run_data']['dag_id']
            latest_run_id = latest_run_data['run_id']
            latest_airflow_link = f"{base_url}/dags/{latest_dag_id}/grid?dag_run_id={latest_run_id}"

            LOG.error("\n=======================================================================")
            LOG.error(
                f"❌ CRITICAL: Found {num_total_failures} failed RUNS ({num_unresolved} unresolved, {num_resolved} resolved) in the last {days_to_check} days.")
            LOG.error(f"🔗 AIRFLOW UI LINK (Latest Failure: {latest_dag_id}): {latest_airflow_link}")
            LOG.error("=======================================================================")

            for index, run_details in enumerate(sorted_all_failures):

                dag_id = run_details['run_data']['dag_id']
                run_id = run_details['run_id']
                is_latest = index == 0
                status_type = run_details['status_type']
                failure_type = run_details['failure_type']
                is_paused = dags_status.get(dag_id)

                airflow_link = f"{base_url}/dags/{dag_id}/grid?dag_run_id={run_id}"

                full_log_content_for_slack = run_details['full_log_content']
                task_details_for_slack = []

                LOG.error(
                    f"\n--- {run_details['header_emoji']} Reporting {status_type} Failure ({failure_type}): DAG '{dag_id}', Run '{run_id}' ---")
                LOG.error(f"  🔗 AIRFLOW UI LINK: {airflow_link}")

                for error in run_details['errors']:
                    task_id = error['task_id']
                    state = error['state']
                    LOG.error(f"  Task ID: {task_id} (State: {state})")
                    task_details_for_slack.append(f"• Task `{task_id}` (State: {state})")

                task_list = "\n".join(task_details_for_slack)
                failed_dt = get_run_datetime(run_details['run_data'], 'end_date')
                fail_time_str = failed_dt.strftime('%Y-%m-%d %H:%M:%S') if failed_dt else "UNKNOWN"

                slack_message = f"""
*{failure_type}*
*DAG:* `{dag_id}`
*Run ID:* `{run_id}`
*Status:* *{status_type}* Failure (at {fail_time_str} UTC)

*🔗 Airflow UI Link:* <{airflow_link} | View in Airflow>

*Failed/Critical Tasks:*
{task_list}
"""
                send_slack_failure_check_notification(slack_conn_id, dag_id, run_id, slack_message,
                                                      full_log_content_for_slack, is_latest, days_to_check, is_paused)
                LOG.info(f"Sent Slack notification for {dag_id} run {run_id} ({status_type}).")

            LOG.error(f"\nTotal failed RUNS reported: {num_total_failures}.")

    else:
        LOG.info(f"✅ No failed runs found in the last {days_to_check} days.")

    return full_runs_report


def analyze_and_upload_concurrency_plot(
        airflow_url: str = None,
        nr_config_path: str = None,
        dag_id: str = "unknown",
        **kwargs):
    """
    Performs Airflow workload concurrency analysis, generates visual reports, and uploads to S3.
    Wrapped with @with_nr_monitoring decorator for telemetry.

    This function queries the Airflow REST API for task instances, calculates minute-by-minute
    active task density, and creates a stacked area chart to identify resource bottlenecks.

    NEW RELIC MONITORING:
    - Isolated Agent Lifecycle: Manually initializes and shuts down the NR agent to ensure
      telemetry capture within the PythonVirtualenvOperator process.
    - Custom Parameters: Tracks 'execution_mode' (virtualenv) and the specific 'airflow_dag'
      triggering the analysis.
    - Performance Alerting: Records 'AirflowConcurrencyPeak' custom event, including
      peak_load, configured_limit, and a boolean flag 'is_over_limit' for NR dashboards/alerts.
    - Resilience: Captures API and S3 upload errors via notice_error and ensures
      data delivery via a 10s shutdown timeout in the 'finally' block.

    Args:
        airflow_url (str, optional): Base URL of the Airflow instance for API requests.
        nr_config_path (str, optional): Path to the New Relic configuration (.ini) file.
        **kwargs:
            - params (dict): Should contain 'days_to_check' (int) and 'max_concurrency_limit' (int).
            - s3_bucket (str): Target S3 bucket for plot storage.
            - dag (DAG): Airflow DAG object provided by provide_context=True.

    Returns:
        dict: Summary containing the S3 presigned URL of the plot, peak load metrics,
              and contribution statistics for Slack notifications.
    """

    from jobs.airflow_pylint_qa.nr_decorators import with_nr_monitoring

    @with_nr_monitoring(task_name="Analyze-Concurrency-Visual")
    def run_analysis(nr_config_path=None, dag_id=None):
        import requests
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')  # Force non-interactive backend
        import matplotlib.pyplot as plt
        import boto3
        import io
        import newrelic.agent
        from datetime import datetime, timedelta, timezone

        # --- 1. Settings ---
        dag_params = kwargs.get('params', {})
        days_back = int(dag_params.get('days_to_check', 1))
        max_limit = int(dag_params.get('max_concurrency_limit', 16))

        bucket_name = kwargs['s3_bucket']
        s3_key = f"qa/plots/concurrency_load_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        # --- 2. Airflow REST API Request ---
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        start_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat().replace('+00:00', 'Z')
        api_url = f"{airflow_url}/api/v1/dags/~/dagRuns/~/taskInstances"

        all_instances = []
        limit, offset = 2000, 0

        print(f"[INFO] Fetching task instances (limit={limit})...")

        while True:
            api_params = {
                "start_date_gte": start_date,
                "state": ["success", "failed", "running"],
                "limit": limit,
                "offset": offset
            }
            response = requests.get(api_url, headers=headers, verify=False, params=api_params, auth=('admin', 'admin'), timeout=120)
            response.raise_for_status()
            data = response.json()

            batch = data.get("task_instances", [])
            all_instances.extend(batch)

            total_entries = data.get("total_entries", 0)
            if offset + limit >= total_entries or not batch:
                break
            offset += limit

        # --- 3. Data Processing ---
        rows = []
        for ti in all_instances:
            if not ti.get('start_date') or not ti.get('end_date'):
                continue
            start, end = pd.to_datetime(ti['start_date']), pd.to_datetime(ti['end_date'])
            curr = start.replace(second=0, microsecond=0)
            while curr <= end:
                rows.append({'timestamp': curr, 'dag_id': ti['dag_id']})
                curr += timedelta(minutes=1)

        if not rows:
            return {"error": f"No task data found for the last {days_back} day(s)."}

        df = pd.DataFrame(rows)
        top_dags = df['dag_id'].value_counts().nlargest(10).index.tolist()
        df['dag_group'] = df['dag_id'].apply(lambda x: x if x in top_dags else 'others')

        pivot_df = df.groupby(['timestamp', 'dag_group']).size().unstack(fill_value=0)
        pivot_df = pivot_df[pivot_df.sum().sort_values(ascending=False).index]

        total_load_series = pivot_df.sum(axis=1)
        current_max_load = int(total_load_series.max())
        peak_time_str = total_load_series.idxmax().strftime('%Y-%m-%d %H:%M UTC')

        # --- 4. Visualization ---
        plt.figure(figsize=(12, 6))
        pivot_df.plot(kind='area', stacked=True, ax=plt.gca(), alpha=0.7, colormap='tab20b')
        plt.axhline(y=max_limit, color='red', linestyle='--', linewidth=2, label=f'Limit ({max_limit})')
        plt.title(f"Airflow Concurrency Analysis - Peak: {current_max_load}")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        plt.ylim(0, max(current_max_load, max_limit) * 1.2)
        plt.tight_layout()

        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', dpi=120)
        img_data.seek(0)
        plt.close('all')

        # --- 5. S3 Upload & Reporting ---
        s3_client = boto3.Session(region_name="us-east-1").client('s3')
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=img_data, ContentType='image/png')

        presigned_url = s3_client.generate_presigned_url(
            'get_object', Params={'Bucket': bucket_name, 'Key': s3_key}, ExpiresIn=86400
        )

        total_points = len(df)
        stats = df['dag_group'].value_counts()
        stats_text = "\n--- DAG GROUPS CONTRIBUTION ---\n"
        for name, count in stats.items():
            percentage = (count / total_points) * 100
            stats_text += f"{name.ljust(45)} | Contribution: {percentage:>5.1f}%\n"

        stats_text = f"Peak Load of {current_max_load} occurred at: {peak_time_str}\n" + stats_text

        # --- 6. New Relic Custom Event ---
        try:
            app = newrelic.agent.application()

            nr_event_data = {
                'peak_load': int(current_max_load),
                'peak_at': str(peak_time_str),
                'configured_limit': int(max_limit),
                'is_over_limit': bool(current_max_load > max_limit),
                'days_checked': int(days_back),
                'total_instances': int(len(all_instances))
            }

            newrelic.agent.record_custom_event(
                'AirflowConcurrencyPeak',
                nr_event_data,
                application=app
            )
            print(f"[NR] Custom event AirflowConcurrencyPeak recorded: {nr_event_data}")

        except Exception as nr_err:
            print(f"[NR ERROR] Failed to record custom event: {nr_err}")

        return {
            "url": presigned_url,
            "peak": current_max_load,
            "limit": max_limit,
            "days": days_back,
            "details_text": stats_text,
            "total_records": len(all_instances)
        }

    return run_analysis(nr_config_path=nr_config_path, dag_id=dag_id)

def prepare_slack_plot_msg(task_id, aws_env, **kwargs):
    """
    Constructs a Block Kit payload with dynamic status and detailed contribution stats.
    Compatible with existing send_slack_report function.
    """
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids=task_id)
    current_env = aws_env

    if not data or "error" in data:
        error_info = data.get("error") if data else "No data received"
        return {"slack_messages": [f":warning: *Concurrency Analysis Error:* {error_info}"], "report_granularity": "single"}

    raw_url = data.get('url', '')
    clean_url = raw_url.strip("', ")

    # --- 1. Dynamic Status Logic ---
    peak = data.get('peak', 0)
    limit = data.get('limit', 16)
    usage_ratio = peak / limit if limit > 0 else 0

    if usage_ratio >= 0.9:
        status_emoji = "🔴"
        status_text = "*CRITICAL: Capacity limit reached*"
    elif usage_ratio >= 0.7:
        status_emoji = "🟡"
        status_text = "*WARNING: High resource usage*"
    else:
        status_emoji = "🟢"
        status_text = "*NORMAL: Healthy environment load*"

    # --- 2. Constructing Blocks ---
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{status_emoji} Airflow Concurrency Report ({current_env})"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{status_text}*\n"
                    f"Received `{data.get('total_records', 0)}` task instances for analysis.\n"
                    f"Analyzed period: last `{data.get('days', 1)}` day(s)."
                )
            }
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Peak Load:*\n`{peak}` concurrent tasks"},
                {"type": "mrkdwn", "text": f"*Environment Limit:*\n`{limit}` tasks"}
            ]
        },
        {
            "type": "image",
            "title": {
                "type": "plain_text",
                "text": "Concurrency Area Chart"
            },
            "image_url": clean_url,
            "alt_text": "Airflow load visualization"
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Top Contributors (Contribution %):*\n```{data.get('details_text', 'No details available')}```"
            }
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Current peak usage is *{int(usage_ratio * 100)}%* of the defined limit."
                }
            ]
        }
    ]

    return {"slack_messages": [blocks], "report_granularity": "single"}


def find_newrelic_config():
    """
    Scans for New Relic configuration file with a global safety net.
    Guaranteed to return either a path or None, never fails the task.
    """
    import os

    try:
        discovered_path = None
        search_paths = [
            os.environ.get('NEW_RELIC_CONFIG_FILE'),
            '/etc/newrelic/newrelic.ini',
            '/opt/airflow/newrelic.ini',
            '/etc/newrelic.ini'
        ]

        print("=== NEW RELIC DIAGNOSTICS ===")

        # 1. Environment Variables check
        nr_env_vars = {k: v for k, v in os.environ.items() if 'NEW_RELIC' in k}
        print(f"[DEBUG] Environment Variables: {nr_env_vars}")

        # 2. File search
        for path in search_paths:
            if path:
                if os.path.exists(path):
                    if not discovered_path:
                        discovered_path = path
                        print(f"[INFO] VALID CONFIG FOUND AT: {discovered_path}")
                else:
                    print(f"[DEBUG] Path does not exist: {path}")

        # 3. Module check
        try:
            import newrelic
            print(f"[DEBUG] New Relic version: {newrelic.version}")
        except ImportError:
            print("[WARNING] New Relic package not found in main environment.")

        print(f"=== DIAGNOSTICS COMPLETE. Returning: {discovered_path} ===")
        return discovered_path

    except Exception as global_err:
        # If anything goes wrong, we log it but don't raise an exception
        print(f"[CRITICAL ERROR] Error during NR discovery: {str(global_err)}")
        print("[INFO] Falling back to None to ensure DAG continuity.")
        return None


# --- DAG Definition ---
@dag(
    schedule_interval=None,
    start_date=days_ago(1),
    default_args=DEFAULT_ARGS,
    params={
        "dag_ids_list": Param(
            default=[],
            description="If provided, only these DAGs will be analyzed. If empty, all active DAGs are analyzed.",
            type="array",
            items={"type": "string"},
            title="List of DAG IDs to analyze (optional)",
        ),
        "report_granularity": Param(
            "single",
            "Determines Slack report format: 'single' (one composite message with full detail) or 'multiple' (one message per DAG with issues).",
            type="string",
            enum=["single", "multiple"],
            title="Slack Report Granularity"
        ),
        "run_import_errors": Param(
            True,
            "If set to False, the Import Errors flow will be skipped entirely.",
            type="boolean",
            title="Run Import Errors Flow",
        ),
        'days_to_check': Param(
            default=1 if AWS_ENV == "L1" else 2,
            type='integer',
            title='Number of days to check',
            description='Number of days to check regression (default 1).',
            minimum=1,
            maximum=7
        ),
        'max_concurrency_limit': Param(
            default=48 if AWS_ENV == "L1" else 16,
            type='integer',
            title='Max Concurrency Limit',
            description='Threshold for the red line and status calculation (default 64).',
            minimum=1,
            maximum=64
        ),
    },
    catchup=False,
    tags=['qa', 'iceberg', 'pylint', 'slack', 'xxVERSION_TAGxx'],
)
def airflow_pylint_qa():
    """
    An automated Airflow QA pipeline that performs Pylint code analysis and mandatory concurrency monitoring across
    all active DAGs within isolated virtual environments, persisting performance metrics into Apache Iceberg
    while delivering regression alerts and import error reports via Slack.
    """
    from jobs.airflow_pylint_qa.pylint_checker import PylintQualityChecker

    env_name = get_airflow_base_url()
    is_debug_mode = False

    find_nr_config_task = PythonOperator(
        task_id='find_newrelic_config',
        python_callable=find_newrelic_config,
    )

    dags_list_xcom = PythonOperator(
        task_id='dags_to_analyze_from_config',
        python_callable=dags_to_analyze_from_config,
    )

    granularity_task = PythonOperator(
        task_id='extract_report_granularity',
        python_callable=extract_report_granularity,
        provide_context=True,
    )

    branch_task = BranchPythonOperator(
        task_id='check_environment_condition',
        python_callable=choose_branch_based_on_env,
        provide_context=True,
    )

    end_dag = EmptyOperator(task_id="all_qa_done", trigger_rule=TriggerRule.NONE_FAILED)

    create_database = AthenaQueryOperator(
        task_id='create_database_if_not_exists',
        query=f"CREATE SCHEMA IF NOT EXISTS {DATABASE}",
        database='default',
        output_location=f"{OUTPUT_LOCATION}/schema/",
        work_group=WORK_GROUP,
    )

    iceberg_ddl_task = PythonOperator(
        task_id='create_iceberg_table_ddl',
        python_callable=create_iceberg_table_ddl,
        provide_context=True,
    )

    create_iceberg_table = AthenaQueryOperator(
        task_id='create_iceberg_table',
        query="{{ task_instance.xcom_pull(task_ids='create_iceberg_table_ddl', key='return_value') }}",
        database=DATABASE,
        output_location=f"{OUTPUT_LOCATION}/table/",
        work_group=WORK_GROUP,
    )

    # --- Failure check
    dags_list_task = PythonOperator(
        task_id='qa_get_active_dags',
        python_callable=get_active_dags_for_failure_check,
        provide_context=True,
    )

    unresolved_failures_check = PythonOperator(
        task_id='qa_check_unresolved_failures',
        python_callable=check_unresolved_failures_task_integrated,
        provide_context=True,
        op_kwargs={
            'dags_list_task_id': 'qa_get_active_dags',
        }
    )

    pylint_results_task = PythonVirtualenvOperator(
        python_version="3.9",
        requirements=[
            'pylint',
            'python-dateutil',
            'requests',
            'newrelic',
        ],
        python_callable=execute_pylint_analysis,
        task_id='pylint_results_task',
        venv_cache_path=Path(f"{VENV_BASE_DIR}/{DAG_NAME}"),
        provide_context=True,
        on_success_callback=DEFAULT_ARGS["on_success_callback"],
        on_failure_callback=DEFAULT_ARGS["on_failure_callback"],
        op_kwargs={
            'nr_config_path': "{{ task_instance.xcom_pull(task_ids='find_newrelic_config') }}",
            'airflow_api_url': get_airflow_base_url(),
            'dag_id': DAG_NAME,
            'is_debug_mode': is_debug_mode,
            'dag_ids_from_config': f"{{{{ task_instance.xcom_pull(task_ids='{dags_list_xcom.task_id}', key='return_value') }}}}",
            'python_path_override': NEW_PYTHONPATH,
        },
        env_vars={
            "NEW_RELIC_AGENT_ENABLED": "true",
            "NEW_RELIC_PYTHON_IGNORE_VERIFY_ENV": "true",
            "NEW_RELIC_APP_NAME": f"Airflow-Pylint-Analysis-{AWS_ENV}",
        },
        system_site_packages=False,
    )

    insert_sql_queries_task = PythonOperator(
        task_id='split_issues_into_sql_batches',
        python_callable=PylintQualityChecker.split_issues_into_sql_batches,
        provide_context=True,
        op_kwargs={'pylint_results_task_id': pylint_results_task.task_id}
    )

    insert_pylint_batch = AthenaQueryOperator.partial(
        task_id='insert_pylint_batch',
        retries=3,
        retry_delay=timedelta(seconds=10),
        database=DATABASE,
        output_location=f"{OUTPUT_LOCATION}/insert/",
        work_group=WORK_GROUP,
    ).expand(query=insert_sql_queries_task.output)  # Dynamic map over the list of queries

    pylint_slack_messages_task = PythonOperator(
        task_id='prepare_slack_messages',
        python_callable=prepare_slack_messages,
        provide_context=True,
        op_kwargs={
            'airflow_api_url': get_airflow_base_url(),
            'pylint_results_task_id': pylint_results_task.task_id,
        }
    )

    pylint_report_task = PythonOperator(
        task_id='send_pylint_summary_report',
        python_callable=send_slack_report,
        provide_context=True,
        op_kwargs={
            'slack_messages_task_id': 'prepare_slack_messages',
        }
    )

    # --- Regression FLOW
    historical_data_task = PythonVirtualenvOperator(
        python_version="3.9",
        requirements=[
            'awswrangler',
            'boto3',
            'pandas',
            'newrelic',
        ],
        python_callable=get_historical_scores,
        task_id='get_historical_scores_task',
        venv_cache_path=Path(f"{VENV_BASE_DIR}/{DAG_NAME}"),
        provide_context=True,
        on_success_callback=DEFAULT_ARGS["on_success_callback"],
        on_failure_callback=DEFAULT_ARGS["on_failure_callback"],
        op_kwargs={
            'nr_config_path': "{{ task_instance.xcom_pull(task_ids='find_newrelic_config') }}",
            'database': DATABASE,
            'table': TABLE,
            'sql_template': ATHENA_N_MINUS_1_RESULTS_SQL_TEMPLATE,
            'output_location': OUTPUT_LOCATION,
            'work_group': WORK_GROUP,
            'dag_ids_from_config': f"{{{{ task_instance.xcom_pull(task_ids='{dags_list_xcom.task_id}', key='return_value') }}}}",
            'dag_id': DAG_NAME,
            'python_path_override': NEW_PYTHONPATH,
        },
        env_vars={
            "NEW_RELIC_AGENT_ENABLED": "true",
            "NEW_RELIC_PYTHON_IGNORE_VERIFY_ENV": "true",
            "NEW_RELIC_APP_NAME": f"Airflow-Pylint-Regression-{AWS_ENV}",
        },
        system_site_packages=False,
    )

    check_for_regression_task = PythonOperator(
        task_id='check_for_regression',
        python_callable=check_for_regression,
        provide_context=True,
        op_kwargs={
            'pylint_results_task_id': pylint_results_task.task_id,
            'historical_data_task_id': historical_data_task.task_id,
        }
    )

    regression_slack_messages_task = PythonOperator(
        task_id='prepare_regression_slack_messages',
        python_callable=prepare_regression_slack_messages,
        provide_context=True,
        op_kwargs={
            'airflow_api_url': get_airflow_base_url(),
            'regression_report_task_id': check_for_regression_task.task_id,
        }
    )

    regression_report_task = PythonOperator(
        task_id="send_regression_alert",
        python_callable=send_slack_report,
        provide_context=True,
        op_kwargs={
            'slack_messages_task_id': regression_slack_messages_task.task_id,
        }
    )

    import_errors_check_task = PythonOperator(
        task_id='qa_check_airflow_import_errors',
        python_callable=check_airflow_import_errors_and_prepare_report,
        op_kwargs={
            'api_url': get_airflow_base_url(),
            'env_name': env_name,
            'is_debug_mode': is_debug_mode,
            'python_path_override': NEW_PYTHONPATH
        }
    )

    process_payload_task = process_slack_payload(
        api_url = get_airflow_base_url(),
    )

    import_errors_report_task = PythonOperator(
        task_id="send_import_errors_notification",
        python_callable=send_slack_report,
        provide_context=True,
        op_kwargs={
            'slack_messages_task_id': 'process_slack_payload_task',
        }
    )

    import_errors_branch_check = BranchPythonOperator(
        task_id='import_errors_branch_check',
        python_callable=determine_import_errors_flow,
        provide_context=True,
    )

    import_errors_skip_signal = EmptyOperator(
        task_id='import_errors_skip_signal',
    )

    # Perform the analysis based on params 'days_to_check' and 'max_concurrency_limit'
    concurrency_analysis = PythonVirtualenvOperator(
        task_id='analyze_concurrency_visual',
        python_version="3.9",
        requirements=['requests', 'pandas', 'matplotlib', 'boto3', 'newrelic'],
        python_callable=analyze_and_upload_concurrency_plot,
        op_kwargs={
            'airflow_url': get_airflow_base_url(),
            's3_bucket': INTEGRATION_TEST_BUCKET,
            'nr_config_path': "{{ ti.xcom_pull(task_ids='find_newrelic_config') }}",
            'dag_id': DAG_NAME
        },
        provide_context=True,
        on_success_callback=DEFAULT_ARGS["on_success_callback"],
        on_failure_callback=DEFAULT_ARGS["on_failure_callback"],
        venv_cache_path=Path(f"{VENV_BASE_DIR}/{DAG_NAME}"),
        env_vars={
            "NEW_RELIC_LICENSE_KEY": "d8bccaaf890065c13cd3d2d218c7745e92884db6",
            "NEW_RELIC_AGENT_ENABLED": "true",
            "NEW_RELIC_PYTHON_IGNORE_VERIFY_ENV": "true",
            "NEW_RELIC_APP_NAME": f"Airflow-Concurrency-Analysis-{AWS_ENV}",
            "NEW_RELIC_ADMIN_COMMAND": "false",
            "NEW_RELIC_STARTUP_TIMEOUT": "10.0",
            "NEW_RELIC_LOG": "stdout",
            "NEW_RELIC_LOG_LEVEL": "debug",
            "NEW_RELIC_STARTUP_DEBUG": "true",
            "NEW_RELIC_ENVIRONMENT": "production",
            "NEW_RELIC_MONITOR_MODE": "true",
            "PYTHONPATH": NEW_PYTHONPATH,
        },
        system_site_packages=False,
    )

    # Prepare the PLOT message
    concurrency_msg_prep = PythonOperator(
        task_id='prepare_concurrency_msg',
        python_callable=prepare_slack_plot_msg,
        op_kwargs={'task_id': 'analyze_concurrency_visual',
                   'aws_env': AWS_ENV},
        provide_context=True,
    )

    # Send report via existing Slack Webhook
    concurrency_report = PythonOperator(
        task_id="send_concurrency_report",
        python_callable=send_slack_report,
        op_kwargs={'slack_messages_task_id': 'prepare_concurrency_msg'},
        provide_context=True,
    )

    cleanup_old_plots = S3DeleteObjectsOperator(
        task_id="cleanup_old_plots",
        bucket=INTEGRATION_TEST_BUCKET,
        prefix="qa/plots/",
        from_datetime=datetime.now(timezone.utc) - timedelta(days=90),
        to_datetime=datetime.now(timezone.utc) - timedelta(days=30),
        aws_conn_id="aws_default",
        verify=False,
    )

    # ==============================================================================
    # 1. DDL Branch
    # ==============================================================================

    [dags_list_xcom, granularity_task] >> find_nr_config_task >> branch_task
    create_database >> iceberg_ddl_task >> create_iceberg_table
    branch_task >> [create_database, end_dag]

    # ==============================================================================
    # 2. Shared dependencies
    # ==============================================================================
    pylint_start_dependency = create_iceberg_table
    pylint_start_dependency >> pylint_results_task
    dags_list_xcom >> historical_data_task
    find_nr_config_task >> [pylint_results_task, historical_data_task]

    # ==============================================================================
    # 3. Regression report FLOW
    # ==============================================================================
    regression_flow = (
            [pylint_results_task, historical_data_task]
            >> check_for_regression_task
            >> regression_slack_messages_task
            >> regression_report_task
    )

    # ==============================================================================
    # 4. PYLINT INSERT FLOW
    # ==============================================================================
    pylint_insert_flow = (
            [pylint_results_task, historical_data_task]
            >> insert_sql_queries_task
            >> insert_pylint_batch
    )

    # ==============================================================================
    # 5. PYLINT Summary report
    # ==============================================================================
    pylint_report_flow = (
            pylint_results_task
            >> pylint_slack_messages_task
            >> pylint_report_task
    )

    # ==============================================================================
    # 6. Import Errors Flow:
    # ==============================================================================

    find_nr_config_task >> import_errors_branch_check
    import_errors_branch_check >> [import_errors_check_task, import_errors_skip_signal]
    import_errors_check_task >> process_payload_task >> import_errors_report_task

    # ==============================================================================
    # 7. FAILURE MONITORING FLOW
    # ==============================================================================

    failure_check_flow = (
            find_nr_config_task >> dags_list_task >> unresolved_failures_check
    )

    # ==============================================================================
    # 8. CONCURRENCY ANALYSIS FLOW
    # ==============================================================================

    find_nr_config_task >> concurrency_analysis >> concurrency_msg_prep >> concurrency_report >> cleanup_old_plots

    # ==============================================================================
    # 9. Completion
    # ==============================================================================
    [
        insert_pylint_batch,
        pylint_report_task,
        regression_report_task,
        import_errors_report_task,
        import_errors_skip_signal,
        unresolved_failures_check,
        cleanup_old_plots
    ] >> end_dag


airflow_pylint_qa()
