import json
import os
import random
import uuid
import boto3
from datetime import datetime, timedelta
from botocore.config import Config

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from confluent_kafka import Producer

# Configuration for LocalStack (Kinesis)
KINESIS_ENDPOINT = "http://gmdh-localstack:4566"
KINESIS_STREAM = "telemetry-stream"
REGION = "us-east-1"

# Boto3 Config to prevent infinite hanging
BOTO_CONFIG = Config(
    connect_timeout=60,
    read_timeout=60,
    retries={'max_attempts': 5}
)

# Configuration for Kafka (Downstream Monitor)
KAFKA_CONF = {
    'bootstrap.servers': 'gmdh-kafka:29092',
    'client.id': 'airflow-market-generator'
}

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

@dag(
    default_args=default_args,
    schedule_interval='*/10 * * * *',
    start_date=datetime(2026, 3, 20),
    catchup=False,
    tags=['production', 'kinesis', 'kafka', 'sp-api', 'cybersource']
)
def market_transaction_generator():

    @task
    def produce_market_events():
        """
        Generates paired events (Amazon SP-API + Cybersource)
        with a shared order_id and pushes them to Kinesis.
        """

        effective_endpoint = os.getenv('KINESIS_ENDPOINT', 'http://gmdh-localstack:4566').strip()

        # Log for debugging
        print(f"DEBUG: Using effective_endpoint: '{effective_endpoint}'")

        # 2. Validity check
        if not effective_endpoint.startswith('http'):
            # If something unexpected arrives, force the default
            effective_endpoint = 'http://gmdh-localstack:4566'

        kinesis = boto3.client(
            'kinesis',
            endpoint_url=effective_endpoint,
            region_name=REGION,
            aws_access_key_id='test',
            aws_secret_access_key='test',
            config=BOTO_CONFIG
        )

        events_sent_count = 0
        all_events = []
        # Generate 4-8 orders (each with 2 events = 8-16 events total)
        num_orders = random.randint(4, 8)

        print(f"🚀 [Inbound] Starting generation of {num_orders} orders ({num_orders * 2} events)...")

        for _ in range(num_orders):
            # Shared correlation key for both systems
            shared_order_id = f"305-{random.randint(1000000, 9999999)}"

            # 1. Amazon SP-API scenario (Business context)
            amz_payload = {
                'order_id': shared_order_id,
                'marketplace_id': 'ATVPDKIKX0DER',
                'order_status': 'Pending',
                'sku': f"GMDH-PROD-{random.randint(100, 500)}",
                'quantity': random.randint(1, 3)
            }

            # 2. Cybersource scenario (Financial context)
            cs_payload = {
                'order_id': shared_order_id,
                'transaction_id': str(uuid.uuid4()),
                'decision': random.choice(['ACCEPT', 'REJECT', 'REVIEW']),
                'amount': round(random.uniform(20.0, 500.0), 2),
                'currency': 'USD',
                'fraud_score': random.randint(0, 100)
            }

            # Send both events to Kinesis
            for source, payload in [('AMAZON_SP_API', amz_payload), ('CYBERSOURCE', cs_payload)]:
                event_envelope = {
                    'metadata': {
                        'event_id': str(uuid.uuid4()),
                        'source_system': source,
                        'ingestion_ts': datetime.utcnow().isoformat(),
                        'schema_version': '2.1'
                    },
                    'payload': payload
                }

                all_events.append(event_envelope)

                try:
                    kinesis.put_record(
                        StreamName=KINESIS_STREAM,
                        Data=json.dumps(event_envelope),
                        # PartitionKey = order_id ensures both events
                        # land in the same shard for Flink processing
                        PartitionKey=shared_order_id
                    )
                    events_sent_count += 1
                except Exception as e:
                    print(f"⚠️ Kinesis write skipped (non-critical): {str(e)[:80]}")

        print(f"✅ Total dispatched to Kinesis: {events_sent_count}/{num_orders * 2} events.")
        return json.dumps(all_events)

    @task
    def sync_to_kafka_monitor(events_json):
        """
        Writes generated transactions to Kafka for downstream consumption
        by fraud_detection_engine + sends health notification.
        """
        try:
            p = Producer(KAFKA_CONF)

            # 1. Write actual transactions to Kafka (primary data path)
            events = json.loads(events_json) if isinstance(events_json, str) else events_json
            for event in events:
                p.produce(
                    'system-monitor',
                    value=json.dumps(event).encode('utf-8'),
                    key=event.get('payload', {}).get('order_id', '').encode('utf-8')
                )
            p.flush(timeout=10)
            print(f"📊 Kafka: {len(events)} transaction events written to system-monitor topic.")
        except Exception as e:
            print(f"⚠️ Kafka notification failed: {e}")

    # Flow definition
    events_json = produce_market_events()
    notify = sync_to_kafka_monitor(events_json)
    trigger_fraud = TriggerDagRunOperator(
        task_id='trigger_fraud_engine',
        trigger_dag_id='fraud_detection_engine',
        wait_for_completion=False,
        trigger_rule='all_success'
    )

    events_json >> notify >> trigger_fraud

# Instantiate the DAG
generator_dag = market_transaction_generator()