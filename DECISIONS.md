# Design Decisions Record

This document records all confirmed design decisions made during development,
including the rationale behind each choice.

---

## 1. Database

### 1.1 選擇 PostgreSQL + pgvector
**決策：** 使用單一 PostgreSQL 實例（pgvector/pgvector:pg16），同時存放原始資料和向量資料。

**原因：**
- 資料規模小（22,363 users / 12,101 items / 198,502 interactions），不需要獨立向量資料庫
- pgvector 支援 HNSW index，cosine similarity 查詢效能足夠
- 減少系統複雜度，少一個服務就少一個故障點

---

### 1.2 DB Schema 設計
```
user(user_id)
interaction(interaction_id, user_id, item_id, timestamp)
item(item_id, category_id1, category_id2, brand_id, price)
brand(brand_id, brand_name)
category(category_id, category)
item_embedding(item_id, model_version, embedding vector(192))
model_version(id, model_version, is_active, note, created_at)
recommendation_log(id, user_id, recommended_items, created_at)
```

**item_embedding 的 192-dim 原因：**
- model 的 item representation = item_emb(64) + category_emb(64) + brand_emb(64) 串接
- 對應 `sid4srec.py` 的 `items_emb()` 方法輸出

**recommendation_log 設計原因：**
- Serving 每次推薦後自動記錄結果，供 `/feedback` endpoint 判斷是否命中
- 同時作為推薦品質監控資料來源

---

## 2. Model

### 2.1 選擇 SID4SRec（SASRec + Diffusion）
**決策：** 使用碩士論文訓練好的 SID4SRec 模型（diffsas-Beauty-0.pt，後改名為 best_model.pt）。

**原因：**
- SASRec 是 sequential recommendation 的強 baseline
- Diffusion 增強 + 2 層對比學習提升表現
- 已有訓練好的權重，可直接部署

### 2.2 User representation 不存 DB
**決策：** User embedding 在 inference 時從 item sequence 動態計算，不存入 DB。

**原因：**
- SASRec 的 user representation 依賴 item sequence，不是靜態向量
- 存 DB 沒有意義，每次 sequence 更新都要重算

### 2.3 Retrain 策略
- **Hyperparameter search 與 retrain 分開**：retrain 使用已知好參數，不重新搜尋
- **Retrain 頻率**：手動觸發（人工判斷需要時）

---

## 3. Docker 架構

### 3.1 完全容器化
**決策：** 所有服務皆以 Docker 容器運行，無本地 Python 環境依賴。

**原因：**
- 環境一致性，避免「我電腦可以跑但 production 不行」的問題
- 方便之後部署到其他機器

### 3.2 Train / Serve 分離
**決策：** train container 和 serve container 分開，共用 `./models/` bind mount。

**原因：**
- Train 很重（GPU + 大量記憶體），不應常駐
- Serve 需要低延遲，不能被 train 佔用資源影響

### 3.3 Train container 使用 profile
**決策：** train service 加上 `profiles: [train]`，預設不啟動。

**原因：**
- 避免 `docker compose up` 時意外啟動 train，浪費資源
- 由 Airflow 透過 `docker compose --profile train run` 觸發

### 3.4 Blue-Green Deployment
**決策：** serve_blue（port 8000）常駐，serve_green（port 8001）更新時才啟動。

**原因：**
- 更新 model 時先跑 green，確認 health check 正常再切流量
- 減少 downtime 風險

**目前實作：** retrain 後直接重啟 serve_blue（因為只更新 model weights，不換 code），Blue-Green 保留供未來 code 更新時使用。

### 3.5 Model 檔案命名
**決策：** 統一使用 `best_model.pt`，retrain 前備份成 `best_model_prev.pt`。

**原因：**
- 統一命名讓所有 container 的 `MODEL_PATH` 環境變數不需要改動
- 保留上一版本作為 rollback 用途
- 避免無限累積歷史版本佔用磁碟

### 3.6 Base image 選擇
**決策：** Train container 使用 `pytorch/pytorch:2.0.0-cuda11.7`。

**原因：** 對應本機 GPU RTX 2060 的 CUDA 11.7，確保相容性。

---

## 4. Serving API

### 4.1 GET vs POST /recommend 分離
**決策：**
- `GET /recommend`：從 DB 查歷史 interaction，適合一般首頁推薦
- `POST /recommend`：直接接收 item sequence，適合新互動後立即推薦

**原因：** 兩種使用情境不同，分開設計讓 client 有彈性選擇。

### 4.2 Model 在 startup 時載入
**決策：** FastAPI lifespan 啟動時 load model 一次，常駐記憶體。

**原因：** 每次 request 重新 load model 延遲太高（數秒），serving 需要低延遲。

### 4.3 feedback vs interaction 分離
**決策：**
- `POST /feedback`：使用者點擊/購買了推薦的商品
- `POST /interaction`：使用者自行搜尋購買（非推薦來源）

**原因：** 兩者職責不同，feedback 用於評估推薦品質，interaction 用於補充訓練資料。

### 4.4 Implicit feedback only
**決策：** 不設計 explicit feedback（如評分），只記錄行為（點擊/購買）。

**原因：** Amazon Beauty 資料集本身只有 implicit feedback，模型設計也是針對 implicit。

---

## 5. Airflow Pipeline

### 5.1 兩個獨立 DAG
**決策：**
- `manual_retrain`：手動觸發，完整 retrain + 重啟 serving
- `monthly_embedding_update`：每月 1 日 02:00 排程，用現有 model 重新產生 item embedding

**原因：**
- Retrain 很重，應由人工判斷時機，不宜自動排程
- Embedding 可以定期更新，不需要完整 retrain（新 item 加入或 category/brand 資料變動時）

### 5.2 使用 BashOperator 而非 DockerOperator
**決策：** 用 BashOperator 執行 `docker compose run` 指令。

**原因：**
- 學習階段優先，BashOperator 讓流程邏輯更清楚，不被 Docker SDK 設定細節卡住
- 之後熟悉後可升級為 DockerOperator

### 5.3 Airflow 獨立 database
**決策：** Airflow 使用獨立的 `airflow` database，不與 `mlops` database 混用。

**原因：** 兩個系統資料分開，避免互相影響，也方便未來各自備份或遷移。

### 5.4 docker compose run 的關鍵參數
```bash
docker compose -f /opt/airflow/project/docker-compose.yml \
  --project-name project_mlops \
  --profile train run --no-deps --rm train \
  python -m scripts.generate_embeddings
```

- `--project-name project_mlops`：確保 train container 加入正確的 Docker network（和 postgres 同一個 network）
- `--no-deps`：不重新啟動已在跑的 postgres 等依賴服務
- `--rm`：執行完後自動移除 container，不留殘留

**原因：** Airflow container 從 `/opt/airflow/project` 執行 docker compose，project name 預設會變成 `project`（不是 `project_mlops`），導致加入錯誤的 network，連不到 postgres。

### 5.5 Model path 使用絕對路徑
**決策：** `.env` 檔定義 `MLOPS_PROJECT_DIR=E:/demo/project_MLOps`，train service volume 使用 `${MLOPS_PROJECT_DIR}/models:/models`。

**原因：** 從 Airflow container 執行 docker compose 時，`./models` 相對路徑會被解析成 container 內的路徑，Docker daemon（在 Windows host）找不到。絕對路徑確保 Windows host 能正確 mount。

---

## 6. Item Embedding 設計

### 6.1 Embedding 來源
**決策：** 使用 `model.items_emb()` 輸出（item + category + brand embedding 串接，192-dim）。

**原因：**
- Forward pass 輸出是「user representation」用於預測下一個 item，不是 item 本身的 embedding
- `items_emb()` 輸出才是存入 DB 供 RAG 使用的 item representation

### 6.2 Item embedding 用途
**決策：** 存入 `item_embedding` table，供未來 RAG explanation system 使用。

**原因：** 目前 serving 用 `full_sort_predict` 直接算 score，不需要從 DB 查 embedding。但 RAG 需要做 similarity search，所以提前存好。

### 6.3 只存 DB 中存在的 item_id
**決策：** `generate_embeddings.py` 先查詢 DB 的有效 item_id，只存那些 item 的 embedding。

**原因：** model 的 item_size（12105）大於 DB 實際存在的 item 數量（12101），多出來的 ID 不在 `item` table，直接 insert 會 FK 違反。
