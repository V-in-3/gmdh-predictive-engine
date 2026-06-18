from typing import List


def dags_to_analyze_from_config(**kwargs) -> List[str]:
    """Retrieves the list of DAG IDs to analyze from the DAG run parameters."""

    ti = kwargs["ti"]
    params = kwargs.get("params", {})
    raw_dag_list = params.get('dag_ids_list')
    if isinstance(raw_dag_list, list):
        dags_to_analyze = raw_dag_list
    else:
        dags_to_analyze = []
        ti.log.warning("Parameter 'dag_ids_list' not found. Returning empty list.")
    return dags_to_analyze