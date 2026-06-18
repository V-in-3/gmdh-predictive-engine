from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class PylintIssue:
    """Dataclass to store detailed Pylint violation information, including the overall score."""
    dag_id: Optional[str]
    check_timestamp: Optional[datetime]
    overall_score: Optional[float]
    issue_type: Optional[str]
    symbol: Optional[str]
    symbol_code: Optional[str]
    message: Optional[str]
    line_number: Optional[int]
    result_of_last_run: Optional[str]
    time_of_last_run: Optional[datetime] = None

    def to_serializable_dict(self) -> Dict[str, Any]:
        """String datetime/Timestamp -> ISO 8601."""
        data = asdict(self)

        if data['check_timestamp'] is not None:
            data['check_timestamp'] = str(data['check_timestamp'])

        if data['time_of_last_run'] is not None:
            data['time_of_last_run'] = str(data['time_of_last_run'])

        return data


def get_historical_scores(
        nr_config_path: str = None,
        database: str = None,
        table:  str = None,
        sql_template:  str = None,
        output_location:  str = None,
        work_group:  str = None,
        dag_ids_from_config: 'Optional[List[str]]' = None,
        dag_id: str = "unknown",
        **kwargs
) -> 'List[Dict[str, Any]]':
    """
    Fetches historical Pylint scores from AWS Athena and prepares them for regression analysis.

    This function performs an asynchronous Athena query using the UNLOAD command for
    high-performance Parquet extraction, parses the results into structured objects,
    and handles data serialization for XCom.

    NEW RELIC MONITORING:
    - Manual Agent Lifecycle: Since this runs in a PythonVirtualenvOperator, the NR agent
      is initialized manually within the task's process.
    - Contextual Tracking: Attaches 'execution_mode' and 'airflow_dag' as custom parameters
      to the NR transaction.
    - Analytics: Records 'PylintAnalysisSummary' custom event with aggregated metrics
      (total issues and average score).
    - Reliability: Implements a mandatory shutdown_agent call in the 'finally' block
      to ensure all telemetry data is flushed before the virtualenv is destroyed.

    Args:
        nr_config_path (str, optional): Path to the New Relic .ini configuration file.
        database (str): Athena database name.
        table (str): Iceberg/Athena table name containing historical results.
        sql_template (str): SQL query template with placeholders for database, table, and filters.
        output_location (str): S3 path where Athena will store query results.
        work_group (str): Athena workgroup for execution control and cost tracking.
        dag_ids_from_config (List[str], optional): List of specific DAGs to filter.
            If None, all active DAGs are queried.
        **kwargs: Airflow context and additional parameters like 'python_path_override'.

    Returns:
        List[Dict[str, Any]]: A list of serialized PylintIssue dictionaries containing
            historical scores and metadata.
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

    @with_nr_monitoring(task_name="Pylint-Historical-Scores-Fetch")
    def run_fetch(nr_config_path=None, dag_id=None):

        import os
        import newrelic.agent

        try:
            import boto3
            import awswrangler as wr
            import pandas as pd
            import time
            import json
            from typing import List
            from datetime import datetime
            from typing import Optional, Dict, Any
            from dataclasses import dataclass, asdict

            @dataclass
            class PylintIssue:
                """Dataclass to store detailed Pylint violation information, including the overall score."""
                dag_id: Optional[str]
                check_timestamp: Optional[datetime]
                overall_score: Optional[float]
                issue_type: Optional[str]
                symbol: Optional[str]
                symbol_code: Optional[str]
                message: Optional[str]
                line_number: Optional[int]
                result_of_last_run: Optional[str]
                time_of_last_run: Optional[datetime] = None

                def to_serializable_dict(self) -> Dict[str, Any]:
                    """String datetime/Timestamp -> ISO 8601."""
                    data = asdict(self)

                    if data['check_timestamp'] is not None:
                        data['check_timestamp'] = str(data['check_timestamp'])

                    if data['time_of_last_run'] is not None:
                        data['time_of_last_run'] = str(data['time_of_last_run'])

                    return data

            global query_execution_id

            def safe_parse_datetime(value: Optional[Any], **kwargs) -> Optional[datetime]:
                if value is None:
                    return None
                try:
                    if isinstance(value, datetime):
                        return value
                    return datetime.fromisoformat(str(value))
                except (ValueError, TypeError):
                    print(f"Unable to parsedatetime: {value}")
                    return None

            def _execute_unload_query(
                    aws_session: boto3.Session,
                    database: str,
                    output_location: str,
                    work_group: str,
                    final_sql: str
            ) -> str:

                athena_client = aws_session.client('athena')

                unload_query = f"""
                    UNLOAD ({final_sql})
                    TO '{output_location}'
                    WITH (
                        format = 'PARQUET'
                    )
                    """
                print("Executing UNLOAD query to ensure PARQUET format.")

                response = athena_client.start_query_execution(
                    QueryString=unload_query,
                    QueryExecutionContext={
                        'Database': database
                    },
                    WorkGroup=work_group
                )
                return response['QueryExecutionId']

            def _wait_for_query(aws_session: boto3.Session, output_location: str, query_execution_id: str) -> Optional[str]:

                athena_client = aws_session.client('athena')

                while True:
                    response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
                    status = response['QueryExecution']['Status']
                    state = status['State']

                    print(f"Query {query_execution_id} status: {state}")

                    if state == 'SUCCEEDED':
                        full_path = f"{output_location.rstrip('/')}/{query_execution_id}/"
                        print(f"Query SUCCEEDED. Returning S3 result path including Query ID: {full_path}")
                        return full_path

                    elif state in ['FAILED', 'CANCELLED']:
                        reason = status.get('StateChangeReason', 'N/A')
                        print(f"Athena query {query_execution_id} FAILED or CANCELLED: {reason}")
                        return None

                    time.sleep(5)

            def execute_and_read_athena_query(boto3_session: boto3.Session, query_execution_id, **kwargs) -> pd.DataFrame:

                print(f"The query is being executed: {query_execution_id}")
                try:
                    df_results = wr.athena.get_query_results(
                        query_execution_id=query_execution_id,
                        boto3_session=boto3_session
                    )
                    print("The data has been successfully loaded into the DataFrame.")
                    print(df_results.head())
                    return df_results
                except Exception as e:
                    print(f"Error loading query results: {e}")

            aws_session = boto3.Session(region_name="us-east-1")
            print(f"Boto3 Session created using Airflow Env: {aws_session.profile_name}")

            def _clear_s3_path(s3_path: str, session: boto3.Session, **kwargs):

                if not s3_path.startswith("s3://"):
                    print(f"Invalid S3 path to clean up: {s3_path}")
                    return

                try:
                    print(f"CLEANUP: Removing the contents of the S3 prefix:{s3_path}")
                    wr.s3.delete_objects(
                        path=s3_path,
                        boto3_session=session,
                        use_threads=True
                    )
                    print(f"CLEANUP: S3 contents of prefix {s3_path} successfully deleted.")
                except Exception as e:
                    print(f"Critical error while trying to clear S3 path {s3_path}: {e}")
                    pass

            _clear_s3_path(output_location, aws_session)

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

            if dag_ids_list:
                print(f"Target DAG IDs for Athena: {dag_ids_list}")
                dag_list_sql = ', '.join([f"'{dag_id.strip()}'" for dag_id in dag_ids_list])
                where_clause = f" dag_id IN ({dag_list_sql})"
                log_message = f"Querying for target DAG IDs: {dag_list_sql}"
            else:
                where_clause = "1 = 1"
                log_message = "Query ALL DAGs in the table. (No DAG IDs provided or invalid format)."

            print(f"INFO - {log_message}")

            final_sql = sql_template.format(
                database=database,
                table=table,
                where_filter=where_clause
            )

            print(log_message)
            print(final_sql)

            historical_results = []

            try:
                query_execution_id = _execute_unload_query(
                    aws_session=aws_session,
                    database=database,
                    output_location=output_location,
                    work_group=work_group,
                    final_sql=final_sql
                )

                s3_result_path = _wait_for_query(
                    aws_session=aws_session,
                    output_location=output_location,
                    query_execution_id=query_execution_id
                )

                if not s3_result_path:
                    print(f"Could not retrieve valid S3 path for query {query_execution_id}. Aborting.")
                    return []
                print(f"{s3_result_path}")

            except Exception as e:
                print(f"An unexpected error occurred during Athena execution: {e}")

            try:
                result_df = execute_and_read_athena_query(aws_session, query_execution_id)

                if result_df is None:
                    print("The execute_and_read_athena_query function returned None. Returning an empty list.")
                    return []

                if result_df.empty:
                    print("The read is successful, but the result is empty (0 rows).")
                else:
                    print(f"Successfully read {len(result_df)} rows.")
                    print(result_df.head())

                historical_results: List[Dict[str, Any]] = result_df.to_dict('records')
                print(f"\nDataFrame converted to {len(historical_results)} dictionaries for iteration.")

            except Exception as e:
                print(f"Error: {e}")

            parsed_issues: List[PylintIssue] = []

            for i, row in enumerate(historical_results):
                try:
                    if not isinstance(row, dict):
                        print(f"Skipping historical row {i}: Expected Dict, got {type(row)}. Data: {row}")
                        continue
                    # print(f"Raw string data {i + 1}: {row}")

                except Exception as e:
                    print(f"FATAL: Error processing historical row {i}: {e}. Skipping row.")
                    continue

                try:
                    issue = PylintIssue(
                        dag_id=row.get('dag_id'),
                        check_timestamp=safe_parse_datetime(row.get('check_timestamp')),
                        overall_score=row.get('overall_score'),
                        issue_type=row.get('issue_type'),
                        symbol=row.get('symbol'),
                        symbol_code=row.get('symbol_code'),
                        message=row.get('message'),
                        line_number=row.get('line_number'),
                        result_of_last_run=row.get('result_of_last_run'),
                        time_of_last_run=safe_parse_datetime(row.get('time_of_last_run'))
                    )
                    parsed_issues.append(issue)
                    # print(f"-> Object created successfully: {issue}")

                except (ValueError, TypeError) as e:
                    print(f"Failed to parse row {i}: Data type error: {e}. Raw Row: {row}")
                except Exception as e:
                    print(f"An unexpected error occurred while processing row {i}: {e}. Raw Row: {row}")

            print(f"\n--- Final Results from Athena (Total: {len(parsed_issues)} issues) ---")
            for issue in parsed_issues:
                print(f"DAG: {issue.dag_id} | Score: {issue.overall_score} | Message: {issue.message[:30]}...")


            try:
                if parsed_issues:
                    avg_score = sum([i.overall_score for i in parsed_issues if i.overall_score]) / len(parsed_issues)
                else:
                    avg_score = 0

                newrelic.agent.record_custom_event('PylintAnalysisSummary', {
                    'dag_id': dag_id,
                    'total_issues': len(parsed_issues),
                    'avg_score': avg_score
                })
            except Exception as nr_err:
                print(f"[NR ERROR] Custom event failed: {nr_err}")

            return [issue.to_serializable_dict() for issue in parsed_issues]

        except Exception as main_err:
            print(f"[CRITICAL] Task failed: {main_err}")
            raise

    return run_fetch(nr_config_path=nr_config_path, dag_id=dag_id)
