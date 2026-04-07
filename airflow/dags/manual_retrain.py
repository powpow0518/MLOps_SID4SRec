"""
DAG: manual_retrain
觸發方式: 手動（Airflow UI 按 Trigger）
功能: 完整 retrain → 驗證 → Blue-Green 切換

Blue-Green 切換流程：
  1. 訓練完成後啟動 serve_green（priority=100，Traefik 自動導流到 GREEN）
  2. 內網直連 health check serve_green，確認新 model 載入正常
  3. 停掉 serve_blue（GREEN 獨自服務，SHORT request 2s grace 足夠）
  4. 最終確認：透過 Traefik 打 /health 確認整條路徑正常
"""

import time
from datetime import datetime

import requests
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

COMPOSE_DIR = "/opt/airflow/project"
TRAEFIK_URL = "http://mlops_traefik:80"       # Traefik 入口（內網）
GREEN_INTERNAL_URL = "http://mlops_serve_green:8000"  # GREEN 內網直連（切流量前確認）


def _wait_healthy(url: str, label: str, attempts: int = 15, interval: int = 5):
    """對指定 URL 的 /health 發送請求，直到回傳 200 或超時。"""
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(f"{url}/health", timeout=10)
            if resp.status_code == 200:
                print(f"[{label}] healthy after {attempt} attempt(s)")
                return
        except Exception as e:
            print(f"[{label}] attempt {attempt}/{attempts} failed: {e}")
        time.sleep(interval)
    raise ValueError(f"[{label}] health check failed after {attempts} attempts")


def health_check_green():
    """切換前確認 serve_green 已正常載入新 model（內網直連）。"""
    _wait_healthy(GREEN_INTERNAL_URL, "serve_green")


def health_check_traefik():
    """切換後確認整條路徑（Client → Traefik → GREEN）正常。"""
    _wait_healthy(TRAEFIK_URL, "traefik→green")


with DAG(
    dag_id="manual_retrain",
    description="手動觸發完整 retrain pipeline（含 Blue-Green 切換）",
    schedule_interval=None,
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
        bash_command=(
            f"docker compose -f {COMPOSE_DIR}/docker-compose.yml "
            "--project-name project_mlops --profile train run --no-deps --rm train "
            "python -m training.train"
        ),
    )

    generate_embeddings = BashOperator(
        task_id="generate_embeddings",
        bash_command=(
            f"docker compose -f {COMPOSE_DIR}/docker-compose.yml "
            "--project-name project_mlops --profile train run --no-deps --rm train "
            "python -m scripts.generate_embeddings"
        ),
    )

    generate_user_representations = BashOperator(
        task_id="generate_user_representations",
        bash_command=(
            f"docker compose -f {COMPOSE_DIR}/docker-compose.yml "
            "--project-name project_mlops --profile train run --no-deps --rm train "
            "python -m scripts.generate_user_representations"
        ),
    )

    # ── Blue-Green 切換 ────────────────────────────────────────────────────────

    start_green = BashOperator(
        task_id="start_green",
        bash_command=(
            f"docker compose -f {COMPOSE_DIR}/docker-compose.yml "
            "--project-name project_mlops --profile green up -d serve_green"
        ),
    )

    # 切流量前：內網直連確認 GREEN 已正常載入新 model
    check_green = PythonOperator(
        task_id="health_check_green",
        python_callable=health_check_green,
    )

    # GREEN 已通過 health check，Traefik priority=100 已搶佔路由
    # 停掉 BLUE（短請求 < 1s，2s 自然 grace period 足夠）
    stop_blue = BashOperator(
        task_id="stop_blue",
        bash_command="sleep 2 && docker stop mlops_serve_blue",
    )

    # 最終確認：透過 Traefik 打一次 /health，確認整條路徑正常
    check_traefik = PythonOperator(
        task_id="health_check_traefik",
        python_callable=health_check_traefik,
    )

    (
        backup_model
        >> run_training
        >> generate_embeddings
        >> generate_user_representations
        >> start_green
        >> check_green
        >> stop_blue
        >> check_traefik
    )
