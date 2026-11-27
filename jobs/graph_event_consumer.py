"""Consume normalized marketplace events and persist graph relationships."""

import json
import os

from jobs.event_normalizer import normalize_risk_event
from jobs.graph_analysis import store_account_relationships


def persist_event(event: dict, connection) -> bool:
    """Persist one event's account relationships."""
    normalized = normalize_risk_event(event)
    account_id = normalized.get("account_id")
    if not account_id:
        return False

    store_account_relationships(
        connection,
        account_id=account_id,
        device_id=normalized.get("device_id"),
        ip_address=normalized.get("ip_address"),
        trial_id=normalized.get("trial_id"),
    )
    return True


def consume_graph_events(max_messages: int = 100, idle_polls: int = 3) -> int:
    """Consume available system-monitor events and persist their graph edges."""
    from confluent_kafka import Consumer
    from jobs.vector_store import VectorStore

    consumer = Consumer({
        "bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "gmdh-kafka:29092"),
        "group.id": os.environ.get("GRAPH_CONSUMER_GROUP", "graph-analysis"),
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([os.environ.get("GRAPH_EVENTS_TOPIC", "system-monitor")])

    store = VectorStore()
    connection = store._get_conn()
    processed = 0
    empty_polls = 0

    try:
        while processed < max_messages and empty_polls < idle_polls:
            message = consumer.poll(1.0)
            if message is None:
                empty_polls += 1
                continue
            if message.error():
                raise RuntimeError(str(message.error()))

            try:
                event = json.loads(message.value().decode("utf-8"))
                if persist_event(event, connection):
                    processed += 1
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                print(f"Skipping invalid graph event: {exc}")
            empty_polls = 0
    finally:
        consumer.close()
        store.close()

    return processed