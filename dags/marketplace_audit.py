import json
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.providers.mysql.hooks.mysql import MySqlHook
from confluent_kafka import Consumer, TopicPartition, Producer

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}


@dag(
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 3, 18),
    catchup=False,
    max_active_runs=1,  # IMPORTANT: only one run at a time
    tags=['audit', 'parallel', 'recursive']
)
def marketplace_audit():
    @task
    def sync_infrastructure():
        mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
        sql = """
              CREATE TABLE IF NOT EXISTS raw_subscriptions
              (
                  id           INT AUTO_INCREMENT PRIMARY KEY,
                  kafka_offset BIGINT UNIQUE,
                  payload      JSON,
                  created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
              ); \
              """
        mysql_hook.run(sql)

    @task
    def get_partition_ids():
        return [0, 1, 2]

    @task
    def consume_partition(partition_id: int, **context):
        run_id = context['run_id']
        BOOTSTRAP_SERVERS = 'kafka-interview-practice-kafka-1:29092'

        # Initialize consumer
        c = Consumer({
            'bootstrap.servers': BOOTSTRAP_SERVERS,
            'group.id': f'audit-scan-{run_id}-{partition_id}',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        })

        # Initialize producer for DLQ
        p = Producer({'bootstrap.servers': BOOTSTRAP_SERVERS})

        try:
            tp = TopicPartition("raw-subscriptions", partition_id, 0)
            c.assign([tp])
            c.seek(tp)

            print(f"🚀 Partition {partition_id}: Starting full scan from offset 0...")

            mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
            processed_count = 0
            empty_polls = 0
            max_empty_polls = 15 # Slightly increased to skip over large gaps

            while empty_polls < max_empty_polls:
                msg = c.poll(2.0)
                if msg is None:
                    empty_polls += 1
                    continue

                if msg.error():
                    print(f"❌ Kafka Error: {msg.error()}")
                    continue

                # --- MESSAGE PROCESSING BLOCK ---
                try:
                    payload_raw = msg.value().decode('utf-8')

                    # 1. Validation: if JSON is malformed, json.loads will throw and we go to except
                    json.loads(payload_raw)

                    # 2. Write to main database (if JSON is valid)
                    sql = "INSERT IGNORE INTO raw_subscriptions (kafka_offset, payload) VALUES (%s, %s)"
                    mysql_hook.run(sql, parameters=(msg.offset(), payload_raw))

                    processed_count += 1
                    empty_polls = 0

                except Exception as e:
                    print(f"⚠️ Error at offset {msg.offset()}: {e}")

                    # 1. Prepare error report for DLQ
                    error_report = {
                        "error_message": str(e),
                        "original_payload": msg.value().decode('utf-8', errors='replace')
                    }

                    # 2. Write to Kafka DLQ topic
                    p.produce('subscriptions_dlq', value=json.dumps(error_report))
                    p.flush()

                    # 3. Write to MySQL DLQ table
                    dlq_sql = "INSERT IGNORE INTO subscriptions_dlq (kafka_offset, payload) VALUES (%s, %s)"
                    mysql_hook.run(dlq_sql, parameters=(msg.offset(), json.dumps(error_report)))

                    print(f"📥 Error saved to DB and Kafka DLQ.")
                    continue

            print(f"🏁 Partition {partition_id} finished. Successfully synced {processed_count} events.")
            return processed_count

        finally:
            c.close()


    @task
    def validate(processed_results):
        from airflow.api.common.trigger_dag import trigger_dag

        mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
        run_total = sum(processed_results)

        # 1. Get High Watermark (maximum possible offset)
        c = Consumer({
            'bootstrap.servers': 'kafka-interview-practice-kafka-1:29092',
            'group.id': f'audit-checker-{datetime.now().timestamp()}'
        })

        kafka_max = 0
        for i in range(3):
            _, high = c.get_watermark_offsets(TopicPartition("raw-subscriptions", i))
            if high > 0:
                kafka_max = max(kafka_max, high - 1)
        c.close()

        # 2. Get actual MAX offset from our MySQL
        sql = """
              SELECT MAX(max_off) FROM (
                                           SELECT MAX(kafka_offset) as max_off FROM raw_subscriptions
                                           UNION
                                           SELECT MAX(kafka_offset) as max_off FROM subscriptions_dlq
                                       ) as combined_offsets \
              """
        db_max = mysql_hook.get_first(sql)[0] or 0


        # 3. Calculate gap between maximums
        gap = kafka_max - db_max

        print(f"📊 Audit Report:")
        print(f"--- Kafka High Offset (expected): {kafka_max}")
        print(f"--- DB Max Offset (actual): {db_max}")
        print(f"--- Processed this run: {run_total}")

        # MAIN STOP LOGIC:
        # If we found nothing this round (run_total == 0) —
        # it means we've probed all offset gaps and there's no new data.
        if run_total == 0 or gap <= 0:
            print("🟢 Success: All available Kafka data is synced to MySQL.")
            return "Synced"

        # If we're still pulling data - trigger recursion
        print(f"⚠️ Gap of {gap} exists and we are still finding data. Retrying...")

        trigger_dag(
            dag_id='marketplace_audit',
            run_id=f"recursive_audit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            conf={},
            execution_date=None
        )
        return f"Retrying. Gap: {gap}"

        # Chain
    infra = sync_infrastructure()
    p_ids = get_partition_ids()
    workers = consume_partition.expand(partition_id=p_ids)

    infra >> workers >> validate(workers)


dag_instance = marketplace_audit()
