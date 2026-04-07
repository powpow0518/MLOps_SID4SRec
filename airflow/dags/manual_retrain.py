"""
DAG: manual_retrain
觸發方式: 手動（Airflow UI 按 Trigger）
功能: 完整 retrain → 驗證 → Blue-Green 切換

Blue-Green 切換流程（Nginx 版，2026-04-07 從 Traefik 改）：
  1. 訓練完成後啟動 serve_green（帶新 model）
  2. 內網直連 health check serve_green，確認新 model 載入正常
  3. Swap nginx upstream（host file: serve_blue:8000 → serve_green:8000）+ nginx -s reload
  4. 停掉 serve_blue（短請求 < 1s，2s grace + nginx graceful reload 足夠）
  5. 最終確認：透過 Nginx 打 /health 確認整條路徑正常

注意：此 DAG 假設目前 active 為 BLUE，僅支援 BLUE→GREEN 單向切換。
若要支援雙向（GREEN→BLUE 下一輪 retrain），需偵測當前 active 並反向 swap。
"""

import time
from datetime import datetime

import requests
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

COMPOSE_DIR = "/opt/airflow/project"
NGINX_URL = "http://mlops_nginx:80"                   # Nginx 入口（內網）
GREEN_INTERNAL_URL = "http://mlops_serve_green:8000"  # GREEN 內網直連（切流量前確認）
NGINX_CONF_HOST_PATH = f"{COMPOSE_DIR}/docker/nginx.conf"  # Airflow 看得到的 host 路徑


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


def health_check_nginx():
    """切換後確認整條路徑（Client → Nginx → GREEN）正常。"""
    _wait_healthy(NGINX_URL, "nginx→green")


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

    # GREEN 已通過 health check → 把 Nginx upstream 從 BLUE 切換到 GREEN
    # 注意：sed 修改的是 host 上的 docker/nginx.conf（bind mount，nginx container 立即看得到）
    #       接著 nginx -s reload 會 graceful 重啟 worker（舊 worker 處理完連線才退出）
    swap_nginx_upstream = BashOperator(
        task_id="swap_nginx_upstream",
        bash_command=(
            f'grep -q "serve_blue:8000" {NGINX_CONF_HOST_PATH} || '
            f'(echo "ERROR: nginx.conf 不在預期 BLUE 狀態，拒絕切換" && exit 1) && '
            f'sed -i "s|serve_blue:8000|serve_green:8000|" {NGINX_CONF_HOST_PATH} && '
            f'docker exec mlops_nginx nginx -s reload'
        ),
    )

    # Nginx 已導流到 GREEN，停掉 BLUE（短請求 < 1s，2s grace 足夠）
    stop_blue = BashOperator(
        task_id="stop_blue",
        bash_command="sleep 2 && docker stop mlops_serve_blue",
    )

    # 最終確認：透過 Nginx 打一次 /health，確認整條路徑正常
    check_nginx = PythonOperator(
        task_id="health_check_nginx",
        python_callable=health_check_nginx,
    )

    (
        backup_model
        >> run_training
        >> generate_embeddings
        >> generate_user_representations
        >> start_green
        >> check_green
        >> swap_nginx_upstream
        >> stop_blue
        >> check_nginx
    )
