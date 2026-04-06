"""
DAG: manual_retrain
觸發方式: 手動（Airflow UI 按 Trigger）
功能: 完整 retrain → 驗證 → 重啟 serve_blue
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import requests

COMPOSE_DIR = "/opt/airflow/project"  # docker-compose.yml 所在路徑（需 volume mount）
SERVE_URL = "http://mlops_serve_blue:8000"

def validate_model():
    """確認新 model 跑起來之後 health check 正常（含重試，等待 container 重啟）"""
    import time
    for attempt in range(10):
        try:
            resp = requests.get(f"{SERVE_URL}/health", timeout=10)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(5)
    raise ValueError(f"Health check failed after 10 attempts")

with DAG(
    dag_id="manual_retrain",
    description="手動觸發完整 retrain pipeline",
    schedule_interval=None,          # 不排程，只能手動觸發
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["retrain", "manual"],
) as dag:

    backup_model = BashOperator(
        task_id="backup_model",
        bash_command=(
            "[ -f /models/best_model.pt ] && "
            "cp /models/best_model.pt /models/best_model_prev.pt || "
            "echo 'No previous model to backup'"
        ),
    )

    run_training = BashOperator(
        task_id="run_training",
        # 假指令：測試 DAG 流程用，真正 retrain 時換成下面的 docker compose 指令
        bash_command="echo 'Simulating training... done'",
        # bash_command=(
        #     f"docker compose -f {COMPOSE_DIR}/docker-compose.yml "
        #     "--project-name project_mlops --profile train run --no-deps --rm train"
        # ),
    )

    # 產生新的 item embedding 並將此 model version 設為 active
    generate_embeddings = BashOperator(
        task_id="generate_embeddings",
        bash_command=(
            f"docker compose -f {COMPOSE_DIR}/docker-compose.yml "
            "--project-name project_mlops --profile train run --no-deps --rm train "
            "python -m scripts.generate_embeddings"
        ),
    )

    # 用新模型 batch 算所有 user representation，存入 DB（~2s）
    generate_user_representations = BashOperator(
        task_id="generate_user_representations",
        bash_command=(
            f"docker compose -f {COMPOSE_DIR}/docker-compose.yml "
            "--project-name project_mlops --profile train run --no-deps --rm train "
            "python -m scripts.generate_user_representations"
            # Note: -m 模式確保 /app 在 sys.path，training module 才能被 import
        ),
    )

    restart_serving = BashOperator(
        task_id="restart_serving",
        bash_command="docker restart mlops_serve_blue",
    )

    health_check = PythonOperator(
        task_id="health_check",
        python_callable=validate_model,
    )

    (
        backup_model
        >> run_training
        >> generate_embeddings
        >> generate_user_representations
        >> restart_serving
        >> health_check
    )
