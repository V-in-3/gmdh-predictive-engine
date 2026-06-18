FROM apache/airflow:2.10.5-python3.11

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         build-essential \
         default-libmysqlclient-dev \
         librdkafka-dev \
         pkg-config \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    confluent-kafka \
    mysqlclient