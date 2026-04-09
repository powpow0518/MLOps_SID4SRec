"""
DAG: rollback
觸發方式: 手動（Airflow UI 按 Trigger）
功能: 手動 Blue-Green 反向切換，用於 manual_retrain 之後想把流量切回上一版

流程：
  1. detect_current_active: 讀 nginx.conf 找出目前 active（source）、要切回的（target）
  2. start_target: 起 target 容器
  3. health_check_target: 確認 target 內網直連 OK
  4. swap_nginx: sed 反向切換 + nginx -s reload
  5. stop_source: 停 source
  6. health_check_nginx: 最終確認 Nginx → target 路徑正常

注意：這隻 DAG 假設 /models/best_model.pt 已經是想要 rollback 到的版本。
      若要 rollback 到 backup_model，需先手動 cp /models/best_model_prev.pt /models/best_model.pt
"""

import time
from datetime import datetime

import requests
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from airflow import DAG

COMPOSE_DIR = "/opt/airflow/project"
NGINX_URL = "http://mlops_nginx:80"
NGINX_CONF_HOST_PATH = f"{COMPOSE_DIR}/docker/nginx.conf"


def _wait_healthy(url: str, label: str, attempts: int = 15, interval: int = 5):
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


def detect_current_active(**context):
    with open(NGINX_CONF_HOST_PATH) as f:
        content = f.read()
    if "serve_blue:8000" in content:
        plan = {"source": "blue", "target": "green"}
    elif "serve_green:8000" in content:
        plan = {"source": "green", "target": "blue"}
    else:
        raise ValueError("nginx.conf 裡找不到 serve_blue:8000 或 serve_green:8000")
    print(f"[rollback] 將從 {plan['source']} 切回 {plan['target']}")
    return plan


def health_check_target(**context):
    plan = context["ti"].xcom_pull(task_ids="detect_current_active")
    target_url = f"http://mlops_serve_{plan['target']}:8000"
    _wait_healthy(target_url, f"serve_{plan['target']}")


def health_check_nginx():
    _wait_healthy(NGINX_URL, "nginx→target")


with DAG(
    dag_id="rollback",
    description="手動 Blue-Green 反向切換（把流量從目前 active 切回另一個 color）",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["rollback", "manual"],
) as dag:

    restore_model_files = BashOperator(
        task_id="restore_model_files",
        bash_command=(
            f"docker compose -f {COMPOSE_DIR}/docker-compose.yml "
            "--project-name project_mlops --profile train run --no-deps --rm train "
            "sh -c '"
            "[ -f /models/best_model_prev.pt ] || (echo ERROR: best_model_prev.pt not found && exit 1) && "
            "[ -f /models/model_args_prev.pkl ] || (echo ERROR: model_args_prev.pkl not found && exit 1) && "
            "cp /models/best_model_prev.pt /models/best_model.pt && "
            "cp /models/model_args_prev.pkl /models/model_args.pkl && "
            "echo Restored model files from prev backup"
            "'"
        ),
    )

    detect_active = PythonOperator(
        task_id="detect_current_active",
        python_callable=detect_current_active,
    )

    start_target = BashOperator(
        task_id="start_target",
        bash_command=(
            "TARGET={{ ti.xcom_pull(task_ids='detect_current_active')['target'] }} && "
            'if [ "$TARGET" = "green" ]; then '
            f"docker compose -f {COMPOSE_DIR}/docker-compose.yml "
            "--project-name project_mlops --profile green up -d serve_green; "
            "else "
            "docker start mlops_serve_blue; "
            "fi"
        ),
    )

    check_target = PythonOperator(
        task_id="health_check_target",
        python_callable=health_check_target,
    )

    swap_nginx = BashOperator(
        task_id="swap_nginx_upstream",
        bash_command=(
            "SOURCE={{ ti.xcom_pull(task_ids='detect_current_active')['source'] }} && "
            "TARGET={{ ti.xcom_pull(task_ids='detect_current_active')['target'] }} && "
            f'grep -q "serve_${{SOURCE}}:8000" {NGINX_CONF_HOST_PATH} || '
            f'(echo "ERROR: nginx.conf 不在預期 ${{SOURCE}} 狀態，拒絕切換" && exit 1) && '
            f'sed -i "s|serve_${{SOURCE}}:8000|serve_${{TARGET}}:8000|" {NGINX_CONF_HOST_PATH} && '
            f"docker exec mlops_nginx nginx -s reload"
        ),
    )

    stop_source = BashOperator(
        task_id="stop_source",
        bash_command=(
            "SOURCE={{ ti.xcom_pull(task_ids='detect_current_active')['source'] }} && "
            "sleep 2 && docker stop mlops_serve_${SOURCE}"
        ),
    )

    check_nginx = PythonOperator(
        task_id="health_check_nginx",
        python_callable=health_check_nginx,
    )

    (
        restore_model_files
        >> detect_active
        >> start_target
        >> check_target
        >> swap_nginx
        >> stop_source
        >> check_nginx
    )
