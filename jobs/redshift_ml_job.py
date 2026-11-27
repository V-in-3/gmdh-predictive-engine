"""Redshift ML benchmark job with AWS and local mock execution modes."""

import logging
import os
from urllib.parse import urlparse

from airflow.exceptions import AirflowSkipException

DATA_DIR = '/opt/airflow/project/data'
REGION = os.environ.get('REDSHIFT_ML_REGION', os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'))


def check_access():
    """Return Redshift configuration or skip when AWS access is unavailable."""
    if os.environ.get('REDSHIFT_ML_MOCK', 'false').lower() == 'true':
        logging.info("Redshift ML mock enabled: using local test configuration.")
        return {'mock': True, 'database': os.environ.get('REDSHIFT_DATABASE', 'dev')}

    import boto3

    cluster_id = os.environ.get('REDSHIFT_CLUSTER_IDENTIFIER')
    workgroup = os.environ.get('REDSHIFT_WORKGROUP_NAME')
    database = os.environ.get('REDSHIFT_DATABASE', 'dev')
    secret_arn = os.environ.get('REDSHIFT_SECRET_ARN')
    iam_role = os.environ.get('REDSHIFT_IAM_ROLE_ARN')
    s3_uri = os.environ.get('REDSHIFT_ML_S3_URI')

    if not (cluster_id or workgroup) or not secret_arn or not iam_role or not s3_uri:
        logging.info("Redshift ML skipped: configuration is incomplete.")
        raise AirflowSkipException("Redshift ML configuration is incomplete")

    try:
        session = boto3.Session(region_name=REGION)
        if session.get_credentials() is None:
            raise RuntimeError("AWS credentials were not found")
        session.client('sts').get_caller_identity()
        session.client('redshift-data').list_statements(MaxResults=1)
    except Exception as exc:
        logging.info("Redshift ML skipped: AWS access is invalid: %s", exc)
        raise AirflowSkipException("Redshift ML AWS access is unavailable") from exc

    return {
        'mock': False,
        'cluster_id': cluster_id,
        'workgroup': workgroup,
        'database': database,
        'secret_arn': secret_arn,
        'iam_role': iam_role,
        's3_uri': s3_uri,
    }


def run_case(redshift_config):
    """Upload the real benchmark and submit the Redshift ML training SQL."""
    if not redshift_config:
        raise AirflowSkipException("Redshift ML configuration is unavailable")

    if redshift_config.get('mock'):
        logging.info(
            "Redshift ML mock completed: statement_id=mock-redshift-ml, "
            "prediction=0.90, decision=BLOCK."
        )
        return {'statement_id': 'mock-redshift-ml', 'prediction': 0.90, 'decision': 'BLOCK'}

    import boto3

    s3_uri = redshift_config['s3_uri'].rstrip('/') + '/fraud_production_50k.csv'
    parsed_uri = urlparse(s3_uri)
    local_path = f'{DATA_DIR}/fraud_production_50k.csv'
    boto3.client('s3', region_name=REGION).upload_file(
        local_path, parsed_uri.netloc, parsed_uri.path.lstrip('/'),
    )
    logging.info("Uploaded benchmark dataset to %s", s3_uri)

    sql = (
        "CREATE TABLE IF NOT EXISTS fraud_ml_benchmark "
        "(semantic_risk FLOAT, velocity_1h FLOAT, proxy_score FLOAT, "
        "amount_deviation FLOAT, is_fraud INTEGER); "
        "TRUNCATE fraud_ml_benchmark; "
        f"COPY fraud_ml_benchmark FROM '{s3_uri}' "
        f"IAM_ROLE '{redshift_config['iam_role']}' CSV IGNOREHEADER 1; "
        "CREATE MODEL fraud_ml_smoke_test "
        "FROM (SELECT semantic_risk, velocity_1h, proxy_score, "
        "amount_deviation, is_fraud FROM fraud_ml_benchmark) "
        "TARGET is_fraud FUNCTION fraud_ml_smoke_predict "
        f"IAM_ROLE '{redshift_config['iam_role']}' AUTO ON;"
    )
    client = boto3.client('redshift-data', region_name=REGION)
    statement_args = {
        'Database': redshift_config['database'],
        'SecretArn': redshift_config['secret_arn'],
        'Sql': sql,
    }
    if redshift_config['cluster_id']:
        statement_args['ClusterIdentifier'] = redshift_config['cluster_id']
    else:
        statement_args['WorkgroupName'] = redshift_config['workgroup']

    try:
        response = client.execute_statement(**statement_args)
        logging.info("Redshift ML benchmark submitted. statement_id=%s", response['Id'])
        return {'statement_id': response['Id'], 'status': 'SUBMITTED'}
    except Exception as exc:
        logging.info("Redshift ML skipped: benchmark job unavailable: %s", exc)
        raise AirflowSkipException("Redshift ML benchmark job is unavailable") from exc


def predict_transaction(transaction):
    """Return a Redshift ML score, or None when the optional signal is unavailable."""
    if os.environ.get('REDSHIFT_ML_MOCK', 'false').lower() == 'true':
        return 0.90

    import boto3

    cluster_id = os.environ.get('REDSHIFT_CLUSTER_IDENTIFIER')
    workgroup = os.environ.get('REDSHIFT_WORKGROUP_NAME')
    secret_arn = os.environ.get('REDSHIFT_SECRET_ARN')
    database = os.environ.get('REDSHIFT_DATABASE', 'dev')
    model_function = os.environ.get('REDSHIFT_ML_FUNCTION', 'fraud_ml_smoke_predict')

    if not (cluster_id or workgroup) or not secret_arn:
        logging.info("Redshift ML prediction unavailable: configuration is incomplete.")
        return None

    statement_args = {
        'Database': database,
        'SecretArn': secret_arn,
        'Sql': (
            f"SELECT {model_function}(CAST({transaction['semantic_risk']} AS FLOAT), "
            f"CAST({transaction['velocity_1h']} AS FLOAT), "
            f"CAST({transaction['proxy_score']} AS FLOAT), "
            f"CAST({transaction['amount_deviation']} AS FLOAT));"
        ),
    }
    if cluster_id:
        statement_args['ClusterIdentifier'] = cluster_id
    else:
        statement_args['WorkgroupName'] = workgroup

    try:
        client = boto3.client('redshift-data', region_name=REGION)
        statement_id = client.execute_statement(**statement_args)['Id']
        status = client.describe_statement(Id=statement_id)
        if status.get('Status') != 'FINISHED':
            logging.info("Redshift ML prediction not ready: status=%s", status.get('Status'))
            return None
        result = client.get_statement_result(Id=statement_id)
        value = result['Records'][0][0]
        return float(next(iter(value.values())))
    except Exception as exc:
        logging.info("Redshift ML prediction unavailable: %s", exc)
        return None
