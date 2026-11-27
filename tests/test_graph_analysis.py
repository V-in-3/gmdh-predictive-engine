import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.event_normalizer import normalize_risk_event
from jobs.graph_analysis import analyze_relationships
from jobs.graph_event_consumer import persist_event


class FakeCursor:
	def __init__(self):
		self.statements = []

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		return False

	def execute(self, statement, parameters=None):
		self.statements.append((statement, parameters))


class FakeConnection:
	def __init__(self):
		self.cursor_instance = FakeCursor()

	def cursor(self):
		return self.cursor_instance

	def commit(self):
		pass


def test_normal_account_has_no_graph_risk():
	result = analyze_relationships({})

	assert result["graph_risk"] == 0.0
	assert result["graph_reasons"] == []


def test_linked_trial_group_has_high_graph_risk():
	result = analyze_relationships({
		"linked_accounts": 4,
		"shared_devices": 1,
		"shared_ips": 1,
		"trial_count_24h": 4,
	})

	assert result["graph_risk"] == 1.0
	assert len(result["graph_reasons"]) == 4


def test_normalize_payment_event_preserves_graph_and_payment_signals():
	result = normalize_risk_event({
		"metadata": {"event_id": "evt-1", "source_system": "CYBERSOURCE"},
		"payload": {
			"order_id": "order-1",
			"account_id": "account-001",
			"device_id": "device-77",
			"ip_address": "203.0.11.3",
			"decision": "REVIEW",
			"fraud_score": 80,
		},
	})

	assert result["source_system"] == "CYBERSOURCE"
	assert result["account_id"] == "account-001"
	assert result["device_id"] == "device-77"
	assert result["payment_risk"] == 0.8
	assert result["payment_decision"] == "REVIEW"


def test_persist_event_writes_account_relationships():
	connection = FakeConnection()

	persisted = persist_event({
		"metadata": {"source_system": "AMAZON_SP_API"},
		"payload": {
			"account_id": "account-001",
			"device_id": "device-77",
			"ip_address": "203.0.11.3",
			"trial_id": "trial-001",
		},
	}, connection)

	assert persisted is True
	writes = [item for item in connection.cursor_instance.statements if item[1]]
	assert len(writes) == 3
