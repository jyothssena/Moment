FROM --platform=linux/amd64 apache/airflow:2.10.4-python3.11
USER root
RUN apt-get update && apt-get install -y gcc g++ && apt-get clean
USER airflow
RUN pip install --no-cache-dir \
    tensorflow-data-validation \
    google-cloud-bigquery>=3.11.0 \
    google-cloud-bigquery-storage>=2.22.0 \
    pyarrow>=12.0.0
