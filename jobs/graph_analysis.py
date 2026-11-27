"""Graph-based behavioral risk signals for account and subscription events."""

from typing import Any


RELATIONSHIP_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS risk_relationships (
	source_id TEXT NOT NULL,
	source_type TEXT NOT NULL,
	relation TEXT NOT NULL,
	target_id TEXT NOT NULL,
	target_type TEXT NOT NULL,
	first_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
	last_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
	PRIMARY KEY (source_id, relation, target_id)
)
"""


def ensure_relationship_table(connection) -> None:
	"""Create the relationship table used by PostgreSQL graph analysis."""
	with connection.cursor() as cursor:
		cursor.execute(RELATIONSHIP_TABLE_SQL)
	connection.commit()


def store_account_relationships(
	connection,
	account_id: str,
	device_id: str | None = None,
	ip_address: str | None = None,
	trial_id: str | None = None,
) -> None:
	"""Upsert graph edges observed in one account event."""
	relationships = [
		(account_id, "account", "USES", device_id, "device"),
		(account_id, "account", "CONNECTED_FROM", ip_address, "ip"),
		(account_id, "account", "STARTED", trial_id, "trial"),
	]

	ensure_relationship_table(connection)
	with connection.cursor() as cursor:
		for source_id, source_type, relation, target_id, target_type in relationships:
			if not target_id:
				continue
			cursor.execute(
				"""
				INSERT INTO risk_relationships
					(source_id, source_type, relation, target_id, target_type)
				VALUES (%s, %s, %s, %s, %s)
				ON CONFLICT (source_id, relation, target_id)
				DO UPDATE SET last_seen = now()
				""",
				(source_id, source_type, relation, target_id, target_type),
			)
	connection.commit()


def load_relationship_context(connection, account_id: str) -> dict[str, int]:
	"""Load graph aggregates for one account from PostgreSQL."""
	ensure_relationship_table(connection)
	with connection.cursor() as cursor:
		cursor.execute(
			"""
			WITH linked_accounts AS (
				SELECT DISTINCT other.source_id AS account_id
				FROM risk_relationships own
				JOIN risk_relationships other
				  ON own.target_id = other.target_id
				 AND own.relation = other.relation
				WHERE own.source_id = %s
				  AND other.source_type = 'account'
				  AND other.source_id <> %s
				  AND own.relation IN ('USES', 'CONNECTED_FROM')
			),
			group_accounts AS (
				SELECT %s AS account_id
				UNION
				SELECT account_id FROM linked_accounts
			)
			SELECT
				(SELECT count(*) FROM linked_accounts),
				(SELECT count(DISTINCT own.target_id)
				   FROM risk_relationships own
				   JOIN risk_relationships other
					 ON other.target_id = own.target_id
					AND other.relation = own.relation
					AND other.source_id <> own.source_id
				  WHERE own.source_id = %s AND own.relation = 'USES'),
				(SELECT count(DISTINCT own.target_id)
				   FROM risk_relationships own
				   JOIN risk_relationships other
					 ON other.target_id = own.target_id
					AND other.relation = own.relation
					AND other.source_id <> own.source_id
				  WHERE own.source_id = %s AND own.relation = 'CONNECTED_FROM'),
				(SELECT count(*)
				   FROM risk_relationships trial
				   JOIN group_accounts group_member
					 ON group_member.account_id = trial.source_id
				  WHERE trial.relation = 'STARTED'
					AND trial.last_seen >= now() - interval '24 hours')
			""",
			(account_id, account_id, account_id, account_id, account_id),
		)
		row = cursor.fetchone()

	return {
		"linked_accounts": int(row[0]),
		"shared_devices": int(row[1]),
		"shared_ips": int(row[2]),
		"trial_count_24h": int(row[3]),
	}


def analyze_relationships(context: dict[str, Any]) -> dict[str, Any]:
	"""Calculate bounded graph risk from precomputed relationship counts."""
	linked_accounts = max(0, int(context.get("linked_accounts", 0)))
	shared_devices = max(0, int(context.get("shared_devices", 0)))
	shared_ips = max(0, int(context.get("shared_ips", 0)))
	trial_count_24h = max(0, int(context.get("trial_count_24h", 0)))

	risk_score = 0.0
	reasons = []

	if linked_accounts >= 3:
		risk_score += 0.35
		reasons.append("Several accounts are linked to the same identity signals")
	if shared_devices >= 1:
		risk_score += 0.25
		reasons.append("The account shares a device with other accounts")
	if shared_ips >= 1:
		risk_score += 0.15
		reasons.append("The account shares an IP address with other accounts")
	if trial_count_24h >= 3:
		risk_score += 0.25
		reasons.append("Several trials were started by the linked group in 24 hours")

	return {
		"graph_risk": round(min(risk_score, 1.0), 4),
		"linked_accounts": linked_accounts,
		"shared_devices": shared_devices,
		"shared_ips": shared_ips,
		"trial_count_24h": trial_count_24h,
		"graph_reasons": reasons,
	}


def analyze_account_graph(transaction: dict[str, Any]) -> dict[str, Any]:
	"""Read graph context from PostgreSQL when an account identity is present."""
	account_id = transaction.get("account_id")
	if not account_id:
		return analyze_relationships(transaction)

	try:
		from jobs.vector_store import VectorStore

		store = VectorStore()
		connection = store._get_conn()
		store_account_relationships(
			connection,
			account_id=account_id,
			device_id=transaction.get("device_id"),
			ip_address=transaction.get("ip_address"),
			trial_id=transaction.get("trial_id"),
		)
		context = load_relationship_context(connection, account_id)
		store.close()
		return analyze_relationships(context)
	except Exception:
		return analyze_relationships(transaction)
