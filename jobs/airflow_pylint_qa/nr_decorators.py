def with_nr_monitoring(task_name):
    """
    Factory decorator for PythonVirtualenvOperator.
    Automatically initializes, creates a transaction, and closes the NR agent.
    """
    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import os
            import newrelic.agent

            nr_config_path = kwargs.get('nr_config_path') or os.environ.get('NEW_RELIC_CONFIG_FILE')
            dag_id = kwargs.get('dag_id', 'unknown')
            nr_initialized = False

            try:
                if nr_config_path and str(nr_config_path) not in ['None', ''] and os.path.exists(nr_config_path):
                    newrelic.agent.initialize(nr_config_path)

                    app_name = os.environ.get("NEW_RELIC_APP_NAME", "Airflow-Venv-Task")
                    newrelic.agent.register_application(name=app_name, timeout=10.0)

                    nr_initialized = True
                    print(f"[NR] Agent initialized for task: {task_name}")
            except Exception as e:
                print(f"[NR ERROR] Initialization failed: {str(e)}")

            if nr_initialized:
                app = newrelic.agent.application()
                newrelic.agent.set_transaction_name(task_name, group="AirflowVenv")

                try:
                    with newrelic.agent.BackgroundTask(app, name=task_name, group="AirflowVenv"):
                        newrelic.agent.add_custom_attribute('task_name', task_name)
                        newrelic.agent.add_custom_attribute('dag_id', dag_id)

                        try:
                            return func(*args, **kwargs)
                        except Exception as e:
                            newrelic.agent.notice_error()
                            raise
                finally:
                    print("[NR] Flushing data and shutting down...")
                    newrelic.agent.shutdown_agent(timeout=20.0)
            else:
                return func(*args, **kwargs)

        return wrapper
    return decorator