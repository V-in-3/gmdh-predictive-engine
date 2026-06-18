import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

import re
import requests
import urllib3
import subprocess
from dateutil import parser as date_parser

PROJECT_ROOT = "/opt/airflow"

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from jobs.airflow_pylint_qa.regression import PylintIssue

ATHENA_INSERT_BATCH_SIZE = 100
ATHENA_DB = "airflow_validation"
ATHENA_TBL = "results"


def check_airflow_path_existence():
    ec2_path = "/home/ec2-user/.local/lib/python3.9/site-packages"
    local_path = "/home/airflow/.local/lib/python3.8/site-packages"

    if os.path.isdir(ec2_path):
        # print(f"ec2_path: {ec2_path}")
        return ec2_path
    elif os.path.isdir(local_path):
        print(f"local_path: {local_path}")
        return local_path
    else:
        default_path = os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
        print(f"default_path: {default_path}")
        return default_path


@dataclass
class PylintQualityChecker:
    dag_folder: str = "/opt/airflow/dags"
    dags_to_ignore: List[str] = field(default_factory=list)
    api_url: str = "http://localhost:8081/api/v1"
    env_name: str = "L3"
    headers: Dict[str, str] = field(
        default_factory=lambda: {"Content-Type": "application/json"}
    )
    is_debug_mode: bool = False
    config_options = None

    # Pylint configuration attributes:
    # C0114(missing-module-docstring),
    # C0115(missing-class-docstring),
    # C0116(missing-function-docstring)
    # C0103(invalid-name),
    # C0301(line too long)
    # C0411(wrong-import-order)
    # W0613(unused-argument),
    # E0611(no-name-in-module) to avoid cogent import errors
    PYLINT_DISABLE: str = "E0611,C0114,C0115,C0116,C0103,C0301,C0411,W0613"

    # Cache to avoid repeated API calls
    dag_run_history_cache: Dict[str, Tuple[str, Optional[datetime]]] = field(default_factory=dict)

    def __post_init__(self):

        self.api_base_url = self.api_url.rstrip('/')

    def _connect_to_api(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[dict]:
        """Connects to the Airflow API and fetches data, disabling SSL verification."""

        full_endpoint = f"{endpoint if endpoint.startswith('api/v1/') else 'api/v1/' + endpoint}"

        url = f"{self.api_base_url}/{full_endpoint}"
        # Suppress warnings for insecure requests
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            # Use 'params' dictionary for pagination or other query parameters
            response = requests.get(url, headers=self.headers, params=params, timeout=10, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            print(f"[ERROR] CRITICAL: Error connecting to Airflow API at {url}. Exception: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(
                f"[ERROR] Failed to fetch data from Airflow API {url}. Status Code: {e.response.status_code if e.response else 'N/A'}. Error: {e}")
            return None

    def fetch_last_completed_run_info(self, dag_id: str) -> Tuple[str, Optional[datetime]]:
        """Retrieves the status and end time of the last successful/failed DAG run."""
        if dag_id in self.dag_run_history_cache:
            return self.dag_run_history_cache[dag_id]

        # Limit to one result, sort by descending execution date
        endpoint = f"api/v1/dags/{dag_id}/dagRuns?limit=1&order_by=-execution_date&state=success&state=failed"
        data = self._connect_to_api(endpoint)

        if data and data.get('dag_runs'):
            latest_run = data['dag_runs'][0]
            status = latest_run.get('state', 'unknown')
            end_date_str = latest_run.get('end_date')

            try:
                end_time = date_parser.parse(end_date_str) if end_date_str else None
            except Exception:
                end_time = None

            result = (status, end_time)
        else:
            # If no history is found
            result = ('no_history', None)

        self.dag_run_history_cache[dag_id] = result
        return result

    def get_all_dags(self) -> List[str]:
        """Retrieves a list of all active DAGs via the Airflow API, handling pagination."""

        all_dags = []
        limit = 100  # Default Airflow API limit
        offset = 0
        total_entries = 1  # Start with 1 to ensure the loop runs at least once

        print("[INFO] Starting fetch of all active DAGs from Airflow API (handling pagination)...")

        while offset < total_entries:
            params = {"limit": limit, "offset": offset}
            data = self._connect_to_api("dags", params=params)

            if data and data.get('dags'):
                # Update total entries on the first run to correctly size the loop
                if offset == 0:
                    total_entries = data.get('total_entries', limit)
                    print(f"[INFO] Total active DAGs reported by API: {total_entries}")

                # Collect DAG IDs, applying filters
                current_dags = [
                    dag['dag_id'] for dag in data['dags']
                    if dag['is_active'] and dag['file_token'] != 'None' and dag['dag_id'] not in self.dags_to_ignore
                ]
                all_dags.extend(current_dags)
                offset += limit

                # If we received fewer DAGs than the limit, and we're past the first page, assume we hit the end
                if len(current_dags) < limit and offset > limit:
                    break
            else:
                # Break loop if no data or no more DAGs are returned
                break

        print(f"[INFO] Found {len(all_dags)} DAGs for analysis after filtering.")
        # Return only the list of valid, active, non-ignored DAG IDs
        return all_dags

    def run_pylint(self, file_path: str) -> Tuple[str, Optional[float]]:
        """
        Executes Pylint with the configured disable codes and returns the output and score.
        """

        SCORE_REGEX = r"rated at ([\d.]+)/10"

        PROJECT_ROOT = "/opt/airflow"
        SITE_PACKAGES_PATH = os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}',
                                          'site-packages')
        os_pathsep = os.pathsep
        env = os.environ.copy()
        current_pythonpath = env.get('PYTHONPATH', '')

        env['PYTHONPATH'] = (
            f"{SITE_PACKAGES_PATH}{os_pathsep}"
            f"{PROJECT_ROOT}{os_pathsep}"
            f"{current_pythonpath}"
        )
        venv_bin_path = os.path.join(sys.prefix, 'bin')
        env['PATH'] = f"{venv_bin_path}{os_pathsep}{env.get('PATH', '')}"

        AIRFLOW_SYSTEM_PATH = check_airflow_path_existence()

        INIT_HOOK = f'import sys; sys.path.append("{AIRFLOW_SYSTEM_PATH}")'

        json_command = [
            'pylint',
            f"--disable={self.PYLINT_DISABLE}",
            file_path,
            "--output-format=json",
            '--reports=n',
            '--init-hook', INIT_HOOK
        ]

        score_command = [
            'pylint', file_path,
            f"--disable={self.PYLINT_DISABLE}",
            '--score=y',
            '--reports=n',
            '--init-hook', INIT_HOOK
        ]

        output: str = ""
        score: Optional[float] = None

        print(f"[INFO] Running Pylint (JSON) command: {' '.join(json_command)}")
        try:
            json_result = subprocess.run(
                json_command, capture_output=True, text=True, check=False, env=env
            )
            output = json_result.stdout

            logging.info(f"--- Pylint JSON Output (Code: {json_result.returncode}) ---")
            if json_result.stderr:
                logging.error(f"Pylint JSON Stderr:\n{json_result.stderr.strip()}")
            if json_result.stdout:
                logging.info(f"Pylint JSON Stdout:\n{json_result.stdout.strip()}")

        except Exception as e:
            logging.error(f"CRITICAL ERROR: Pylint JSON execution failed: {e}")
            return "", None

        print(f"[INFO] Running Pylint (Score) command: {' '.join(score_command)}")
        try:
            score_result = subprocess.run(
                score_command, capture_output=True, text=True, check=False, env=env
            )

            full_score_output = score_result.stdout + score_result.stderr
            score_match = re.search(SCORE_REGEX, full_score_output, re.MULTILINE)

            if score_match:
                score = float(score_match.group(1))
            else:
                print(f"[WARNING] Failed to extract Pylint score (text output) for {file_path}. Setting to None.")
                score = None

            logging.info(f"--- Pylint Score Extracted: {score} ---")

        except Exception as e:
            logging.error(f"CRITICAL ERROR: Pylint Score execution failed: {e}")
            score = None

        return output, score

    def parse_pylint_output(self,
                            output: str,
                            dag_id: str,
                            timestamp: datetime,
                            overall_score: Optional[float]
                            ) -> List[PylintIssue]:
        """
        Parses the Pylint output into a structured list of PylintIssue objects.
        """
        issues = []

        json_start = output.find('[')
        json_end = output.rfind(']')

        if json_start == -1 or json_end == -1:
            print("[ERROR] Could not find valid JSON array in Pylint output. Skipping issue parsing.")
            return issues

        json_output = output[json_start: json_end + 1]

        try:
            messages: List[dict] = json.loads(json_output)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to decode Pylint JSON output: {e}. Raw JSON: {json_output[:200]}...")
            return issues

        last_run_status, last_run_time = self.fetch_last_completed_run_info(dag_id)

        for msg in messages:
            full_symbol_code = msg.get('message-id', 'N/A')
            issue_type_code = msg.get('type', 'U')[0].upper()

            issue = PylintIssue(
                dag_id=dag_id,
                check_timestamp=timestamp,
                overall_score=overall_score,
                issue_type=issue_type_code,
                symbol=msg.get('symbol', 'unknown'),
                symbol_code=full_symbol_code,
                message=msg.get('message', 'N/A'),
                line_number=msg.get('line', 0),
                result_of_last_run=last_run_status,
                time_of_last_run=last_run_time
            )
            issues.append(issue)

        return issues

    def check_import_errors(self) -> Optional[Dict[str, Any]]:

        endpoint = "api/v1/importErrors"
        limit = 100
        offset = 0
        params = {"limit": limit, "offset": offset}
        env_name = self.env_name

        check_url = f"{self.api_base_url}/{endpoint}?limit={limit}"

        print(f"Running validation check on Airflow {env_name} ({check_url})")

        data = self._connect_to_api(endpoint, params=params)

        if data is None:
            message_text = f"CRITICAL FAILURE: Airflow API connection failed for *{env_name}*. Check logs for details. API Endpoint: <{check_url}|Airflow UI>"
            color = "danger"
            return {"message": message_text, "color": color, "granularity": "single"}

        import_errors = data.get('import_errors', [])
        error_count = len(import_errors)

        if error_count == 0:
            success_link_slack = f"<{check_url}|Airflow {env_name} UI>"
            message_text = f"Status check for {success_link_slack} SUCCESSFUL. Broken DAGs: 0."
            print(f"\033[0;32m{message_text}\033[0m")
            return {"message": message_text, "color": "success", "granularity": "single"}
        else:
            unique_files = sorted(list(set(err.get('filename') for err in import_errors if err.get('filename'))))
            total_unique_broken_files = len(unique_files)
            broken_files_list = "\n".join(f"• `{f}`" for f in unique_files[:5])

            airflow_link_slack = f"<{check_url}|Go to Airflow {env_name} Dashboard>"
            message_text = f"⛈️ *ATTENTION: IMPORT ERRORS* | {airflow_link_slack}\n\nFound *{error_count}* broken DAGs (in {total_unique_broken_files} unique files)."

            if broken_files_list:
                message_text += f"\n\n*Example Broken Files:*\n{broken_files_list}"
                if total_unique_broken_files > 5: message_text += "\n(...)"

            color = "danger"
            print(f"\033[0;31m{message_text}\033[0m")

            return {"message": message_text, "color": color, "granularity": "single"}

    @staticmethod
    def _format_issue_value(value: Any) -> str:
        """Helper to safely format single quotes and handle None/NULL."""
        if value is None:
            return "NULL"
        if isinstance(value, str):
            safe_content = value.replace("'", "''")
            return f"'{safe_content}'"
        return str(value)

    @staticmethod
    def build_single_batch_insert_sql(
            issues: List[Dict[str, Any]],
            database: str,
            table: str
    ) -> str:
        """
        Creates an INSERT SQL query for a single batch of found issues.
        """

        if not issues:
            return "SELECT 1 WHERE 1 = 0"

        value_list = []
        for issue_dict in issues:
            sql_safe_dag_id = PylintQualityChecker._format_issue_value(issue_dict.get('dag_id'))
            sql_safe_message = PylintQualityChecker._format_issue_value(issue_dict.get('message'))
            sql_safe_symbol = PylintQualityChecker._format_issue_value(issue_dict.get('symbol'))
            sql_safe_symbol_code = PylintQualityChecker._format_issue_value(issue_dict.get('symbol_code'))
            sql_safe_issue_type = PylintQualityChecker._format_issue_value(issue_dict.get('issue_type'))
            sql_safe_result_of_last_run = PylintQualityChecker._format_issue_value(issue_dict.get('result_of_last_run'))

            score = issue_dict.get('overall_score')
            score_val = str(score) if score is not None else "NULL"
            line_number = issue_dict.get('line_number', 'NULL')

            check_time = issue_dict.get('check_timestamp')
            check_time_str = (
                f"TIMESTAMP '{check_time.strftime('%Y-%m-%d %H:%M:%S')}'"
                if isinstance(check_time, datetime) else "CAST(NULL AS TIMESTAMP)"
            )

            time_of_last_run = issue_dict.get('time_of_last_run')
            time_str = (
                f"TIMESTAMP '{time_of_last_run.strftime('%Y-%m-%d %H:%M:%S')}'"
                if isinstance(time_of_last_run, datetime) else "CAST(NULL AS TIMESTAMP)"
            )

            value = (
                f"({sql_safe_dag_id}, "
                f"{check_time_str}, "
                f"{score_val}, "
                f"{sql_safe_issue_type}, "
                f"{sql_safe_symbol}, "
                f"{sql_safe_symbol_code}, "
                f"{sql_safe_message}, "
                f"{line_number}, "
                f"{sql_safe_result_of_last_run}, "
                f"{time_str})"
            )
            value_list.append(value)

        values_str = ", ".join(value_list)

        sql = f"""
            INSERT INTO {database}.{table}
            (dag_id, check_timestamp, overall_score, issue_type, symbol, symbol_code, message, line_number, result_of_last_run, time_of_last_run)
            VALUES {values_str};
        """
        return sql

    @staticmethod
    def split_issues_into_sql_batches(
            pylint_results_task_id: str,
            **kwargs
    ) -> List[str]:
        """
        Splits the large list of issues into smaller batches (ATHENA_INSERT_BATCH_SIZE max)
        and generates an individual SQL INSERT query for each batch.
        """
        LOG = logging.getLogger(__name__)
        LOG.setLevel(logging.DEBUG)

        ti = kwargs['ti']
        pylint_results: dict = ti.xcom_pull(task_ids=pylint_results_task_id, key='return_value')

        if pylint_results is None:
            LOG.error(f"XCom pull from task '{pylint_results_task_id}' returned None.")
            return ["SELECT 1 WHERE 1 = 0 AND 1 = 2 /* XCom failed to retrieve Pylint results */"]

        all_issues: List[Dict[str, Any]] = pylint_results.get('all_issues', [])

        total_issues = len(all_issues)
        sql_batches: List[str] = []

        LOG.debug(f"Total Pylint issues found: {total_issues}. Splitting into batches of {ATHENA_INSERT_BATCH_SIZE}.")

        if total_issues == 0:
            LOG.info("No Pylint issues found. Returning empty SQL batch.")
            return ["SELECT 1 WHERE 1 = 0"]

        # Loop through the list, stepping by BATCH_SIZE
        for i in range(0, total_issues, ATHENA_INSERT_BATCH_SIZE):
            batch = all_issues[i:i + ATHENA_INSERT_BATCH_SIZE]
            batch_start = i + 1
            batch_end = min(i + ATHENA_INSERT_BATCH_SIZE, total_issues)

            sql = PylintQualityChecker.build_single_batch_insert_sql(
                batch,
                database=ATHENA_DB,
                table=ATHENA_TBL
            )
            sql_batches.append(sql)

            LOG.debug(f"Batch {len(sql_batches)} created: rows {batch_start}-{batch_end}.")

        return sql_batches
