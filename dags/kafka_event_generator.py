import json
import random
import uuid
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from confluent_kafka import Producer

default_args = {
    'owner': 'airflow-admin',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

@dag(
    default_args=default_args,
    schedule_interval='*/5 * * * *', # Every 5 minutes
    start_date=datetime(2026, 3, 18),
    catchup=False,
    tags=['test', 'generator', 'independent']
)
def kafka_event_generator():

    @task
    def produce_random_events():
        """Generates a random number of events of various types"""

        p = Producer({
            'bootstrap.servers': 'gmdh-kafka:29092',
            'client.id': 'airflow-random-generator',
            'acks': 'all'
        })

        topic = "raw-subscriptions"

        # List of possible plans and event types
        plans = ['basic', 'premium', 'ultimate', 'trial']
        event_types = ['SUBSCRIPTION_CREATED', 'PAYMENT_SUCCESS', 'PLAN_UPGRADE']

        # Random number of events for this run
        num_events = random.randint(1, 15)

        print(f"🎲 [GENERATOR] Decided to send {num_events} random events...")

        for _ in range(num_events):
            event_type = random.choice(event_types)

            event_data = {
                'event_id': str(uuid.uuid4()),
                'event_type': event_type,
                'user_id': random.randint(100, 999),
                'plan': random.choice(plans),
                'amount': random.uniform(9.99, 49.99) if event_type == 'PAYMENT_SUCCESS' else 0,
                'timestamp': datetime.utcnow().isoformat()
            }

            def delivery_report(err, msg):
                if err is not None:
                    print(f"❌ Failed: {err}")
                else:
                # Log event type for visibility in Airflow
                    print(f"✅ Sent {event_type} to offset {msg.offset()}")

            p.produce(
                topic,
                value=json.dumps(event_data).encode('utf-8'),
                callback=delivery_report
            )

        p.flush()
        return f"Produced {num_events} diverse events."

    produce_random_events()

generator_dag = kafka_event_generator()