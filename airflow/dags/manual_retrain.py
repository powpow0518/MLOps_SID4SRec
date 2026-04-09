"""
DAG: manual_retrain
觸發方式: 手動（Airflow UI 按 Trigger，可選 dag_run.conf 傳入 epochs / train_batch_size）
功能: 完整 retrain → 驗證 → Blue-Green 切換（雙向，含 auto rollback）

Blue-Green 切換流程（Nginx 版，雙向）：
  1. detect_current_active: 讀 nginx.conf，判斷現在哪個 color 在服務（source / target）
  2. 訓練完成後啟動 target（green 用 compose profile，blue 用 docker start）
  3. 內網直連 health check target，確認新 model 載入正常
  4. Swap nginx upstream（sed 把 source 換成 target）+ nginx -s reload
  5. 停掉 source
  6. 最終確認：透過 Nginx 打 /health 確認整條路徑正常
  7. 若 health_check_nginx 失敗 → rollback_on_failure 自動反向 swap + 重啟 source

可選 dag_run.conf:
  {"epochs": 5, "train_batch_size": 32}  ← 輕量 end-to-end 驗證用
  預設 epochs=1000 / train_batch_size=256（config.py 的 Beauty 最佳值）
"""

import os
import time
from datetime import datetime

import psycopg2
import requests
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

from airflow import DAG

COMPOSE_DIR = "/opt/airflow/project"
NGINX_URL = "http://mlops_nginx:80"
NGINX_CONF_HOST_PATH = f"{COMPOSE_DIR}/docker/nginx.conf"


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


def create_training_snapshot(**context):
    """記錄目前 item / interaction 的 max id，寫入 training_snapshot 表。

    後續所有 task（train、generate_embeddings、generate_user_representations）
    都依照這個 snapshot 做 vocab 凍結，確保 ingest 不會污染訓練期間的資料邊界。
    """
    db_url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(item_id), 0) FROM item")
            max_item_id = cur.fetchone()[0]
            cur.execute("SELECT COALESCE(MAX(interaction_id), 0) FROM interaction")
            max_interaction_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO training_snapshot (max_item_id, max_interaction_id)
                VALUES (%s, %s) RETURNING id
                """,
                (max_item_id, max_interaction_id),
            )
            snapshot_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()

    print(
        f"[create_snapshot] id={snapshot_id} "
        f"max_item_id={max_item_id} max_interaction_id={max_interaction_id}"
    )
    return {
        "snapshot_id": snapshot_id,
        "max_item_id": max_item_id,
        "max_interaction_id": max_interaction_id,
    }


def detect_current_active(**context):
    """讀 nginx.conf 判斷目前 active 的 color，推到 XCom 供下游用。"""
    with open(NGINX_CONF_HOST_PATH) as f:
        content = f.read()
    if "serve_blue:8000" in content:
        plan = {"source": "blue", "target": "green"}
    elif "serve_green:8000" in content:
        plan = {"source": "green", "target": "blue"}
    else:
        raise ValueError("nginx.conf 裡找不到 serve_blue:8000 或 serve_green:8000，無法判斷目前 active")
    print(f"[detect_current_active] source={plan['source']}, target={plan['target']}")
    return plan


def health_check_target(**context):
    """切換前確認 target 內網直連已正常載入新 model。"""
    plan = context["ti"].xcom_pull(task_ids="detect_current_active")
    target_url = f"http://mlops_serve_{plan['target']}:8000"
    _wait_healthy(target_url, f"serve_{plan['target']}")


def health_check_nginx():
    """切換後確認整條路徑（Client → Nginx → target）正常。"""
    _wait_healthy(NGINX_URL, "nginx→target")


with DAG(
    dag_id="manual_retrain",
    description="手動觸發完整 retrain pipeline（含 Blue-Green 雙向切換 + auto rollback）",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["retrain", "manual"],
    params={
        "epochs": Param(1000, type="integer", title="Epochs",
                        description="訓練輪數（輕量驗證可填 5）"),
        "train_batch_size": Param(256, type="integer", title="Train batch size",
                                  description="訓練 batch size（預設 256）"),
    },
) as dag:

    snapshot_task = PythonOperator(
        task_id="create_snapshot",
        python_callable=create_training_snapshot,
    )

    backup_model = BashOperator(
        task_id="backup_model",
        bash_command=(
            "[ -f /models/best_model.pt ] && "
            "cp /models/best_model.pt /models/best_model_prev.pt || "
            "echo 'No previous model to backup' ; "
            "[ -f /models/model_args.pkl ] && "
            "cp /models/model_args.pkl /models/model_args_prev.pkl || "
            "echo 'No previous model_args to backup'"
        ),
    )

    # 訓練：從 DB 拉資料（--use_db），帶入 snapshot 邊界確保 vocab 凍結
    run_training = BashOperator(
        task_id="run_training",
        bash_command=(
            f"docker compose -f {COMPOSE_DIR}/docker-compose.yml "
            "--project-name project_mlops --profile train run --no-deps --rm train "
            "python -m training.train --use_db "
            "--snapshot_item_id {{ ti.xcom_pull(task_ids='create_snapshot')['max_item_id'] }} "
            "--snapshot_interaction_id {{ ti.xcom_pull(task_ids='create_snapshot')['max_interaction_id'] }} "
            "--epochs {{ params.epochs }} "
            "--train_batch_size {{ params.train_batch_size }}"
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

    # ── Blue-Green 雙向切換 ─────────────────────────────────────────────────

    detect_active = PythonOperator(
        task_id="detect_current_active",
        python_callable=detect_current_active,
    )

    # 起 target：green 要用 compose profile 創建；blue 只需 docker start
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

    # sed 把 source 換成 target：先 grep 防呆，確認目前真的是 source
    swap_nginx_upstream = BashOperator(
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

    # ── Auto rollback：health_check_nginx 失敗時觸發 ────────────────────────
    # trigger_rule=ONE_FAILED：只要上游任何一個失敗就執行
    # 反向 swap + 重啟 source，把流量拉回切換前的狀態
    rollback_on_failure = BashOperator(
        task_id="rollback_on_failure",
        trigger_rule=TriggerRule.ONE_FAILED,
        bash_command=(
            "SOURCE={{ ti.xcom_pull(task_ids='detect_current_active')['source'] }} && "
            "TARGET={{ ti.xcom_pull(task_ids='detect_current_active')['target'] }} && "
            f'echo "[ROLLBACK] restoring ${{SOURCE}} as active" && '
            f"docker start mlops_serve_${{SOURCE}} && "
            f'sed -i "s|serve_${{TARGET}}:8000|serve_${{SOURCE}}:8000|" {NGINX_CONF_HOST_PATH} && '
            f"docker exec mlops_nginx nginx -s reload && "
            f'echo "[ROLLBACK] nginx restored to ${{SOURCE}}"'
        ),
    )

    (
        snapshot_task
        >> backup_model
        >> run_training
        >> generate_embeddings
        >> generate_user_representations
        >> detect_active
        >> start_target
        >> check_target
        >> swap_nginx_upstream
        >> stop_source
        >> check_nginx
        >> rollback_on_failure
    )
