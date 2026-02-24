FROM apache/airflow:2.10.4-python3.11

USER root
RUN apt-get update && apt-get install -y gcc g++ && apt-get clean

USER airflow
RUN pip install --no-cache-dir tensorflow-data-validation
