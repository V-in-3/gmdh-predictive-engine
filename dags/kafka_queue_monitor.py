from airflow.decorators import dag, task
from airflow.providers.mysql.hooks.mysql import MySqlHook
from confluent_kafka import Consumer, TopicPartition, KafkaException
from datetime import datetime
import time

# Configuration (using internal Kafka network port)
BOOTSTRAP_SERVERS = 'kafka-interview-practice-kafka-1:29092'
TOPIC = "raw-subscriptions"
MYSQL_CONN_ID = 'mysql_default'

default_args = {
    'owner': 'airflow-admin',
    'retries': 0,
}

@dag(
    dag_id='kafka_lag_monitor',
    default_args=default_args,
    schedule_interval='*/3 * * * *',
    start_date=datetime(2026, 3, 18),
    catchup=False,
    tags=['monitoring', 'infra']
)
def kafka_lag_monitor():

    @task
    def check_sync_gap(threshold=5):
        print(f"🚀 Starting lag check for topic: {TOPIC}")

        # 1. Kafka configuration with strict timeouts
        conf = {
            'bootstrap.servers': BOOTSTRAP_SERVERS,
            'group.id': 'lag-monitor-metadata-only',
            'socket.timeout.ms': 5000,      # 5 sec per network request
            'request.timeout.ms': 5000,     # 5 sec for broker response
            'session.timeout.ms': 6000,
            'enable.partition.eof': False
        }

        c = Consumer(conf)
        kafka_total_events = 0

        try:
            # Check High Watermark for partitions 0, 1, 2
            for i in range(3):
                tp = TopicPartition(TOPIC, i)
                # Use timeout to avoid hanging if partition doesn't exist
                low, high = c.get_watermark_offsets(tp, timeout=3.0)
                if high < 0: # Partition not found
                    print(f"⚠️ Partition {i} not found or unavailable")
                    continue
                kafka_total_events += high
                print(f"📍 Partition {i} high watermark: {high}")
        except KafkaException as ke:
            print(f"❌ Kafka error: {ke}")
            return False # Abort to avoid triggering audit on error
        finally:
            c.close()

        # 2. Get database state
        print("🔍 Querying MySQL...")
        mysql_hook = MySqlHook(mysql_conn_id=MYSQL_CONN_ID)

        try:
            # Get record count.
            # For large tables, SELECT COUNT(*) may hang MySQL.
            result = mysql_hook.get_first("SELECT COUNT(*) FROM raw_subscriptions")
            db_count = result[0] if result else 0
        except Exception as e:
            print(f"❌ MySQL error: {e}")
            return False

        gap = kafka_total_events - db_count

        print(f"📊 --- SYNC STATUS AT {datetime.now().strftime('%H:%M:%S')} ---")
        print(f"--- Kafka Total (High Watermark sum): {kafka_total_events}")
        print(f"--- DB Record Count: {db_count}")
        print(f"--- Current Gap: {gap}")
        print(f"--- Threshold: {threshold}")

        if gap >= threshold:
            print(f"🔥 LAG DETECTED! Gap {gap} exceeds threshold {threshold}")
            return True

        print("🟢 System is in sync. No audit needed.")
        return False

    @task
    def trigger_audit_if_needed(should_run):
        if should_run:
            from airflow.api.common.trigger_dag import trigger_dag
            print("🚀 Triggering main DAG 'marketplace_audit'...")
            try:
                trigger_dag(
                    dag_id='marketplace_audit',
                    run_id=f"monitor_triggered_{int(time.time())}"
                )
            except Exception as e:
                print(f"❌ Failed to trigger audit: {e}")
        else:
            print("💤 Skipping audit trigger.")

    # Execution logic
    is_lagging = check_sync_gap(threshold=5)
    trigger_audit_if_needed(is_lagging)

monitor_dag = kafka_lag_monitor()