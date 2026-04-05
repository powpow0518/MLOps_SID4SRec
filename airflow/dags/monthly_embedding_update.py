"""
DAG: monthly_embedding_update
觸發方式: 每月 1 日 02:00
功能: 用現有 model weights 重新產生 item embedding → 更新 DB → 重啟 serve_blue
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import requests

COMPOSE_DIR = "/opt/airflow/project"
SERVE_URL = "http://mlops_serve_blue:8000"

def validate_serving():
    import time
    for attempt in range(10):
        try:
            resp = requests.get(f"{SERVE_URL}/health", timeout=10)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(5)
    raise ValueError("Health check failed after 10 attempts")

with DAG(
    dag_id="monthly_embedding_update",
    description="每月重新產生 item embedding 並更新 serving",
    schedule_interval="0 2 1 * *",   # 每月 1 日 02:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["embedding", "scheduled"],
) as dag:

    generate_embeddings = BashOperator(
        task_id="generate_embeddings",
        bash_command=(
            f"docker compose -f {COMPOSE_DIR}/docker-compose.yml "
            "--project-name project_mlops --profile train run --no-deps --rm train "
            "python -m scripts.generate_embeddings"
        ),
    )

    restart_serving = BashOperator(
        task_id="restart_serving",
        bash_command="docker restart mlops_serve_blue",
    )

    health_check = PythonOperator(
        task_id="health_check",
        python_callable=validate_serving,
    )

    generate_embeddings >> restart_serving >> health_check
