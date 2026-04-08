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

### 2.0 Train / Inference 流程概覽

**訓練階段**
```
interaction table（user → item sequence）
  + item / brand / category mapping
          ↓
       Model
          ↓
    best_model.pt → /models/
```

**Inference 階段（GET /recommend）**
```
user_id → interaction table → item sequence
          ↓
       Model（full_sort_predict）
          ↓
    user representation（192-dim, seq 最後一位的 output）
          ↓
    scores = user_repr · all_items_emb.T
          ↓
    top-5 item_ids → recommendation_log
                   → user_representation table（UPSERT）
```

**每月 embedding 更新（monthly_embedding_update DAG）**
```
Model（items_emb()：item + category + brand embedding 串接）
          ↓
    item_embedding table（192-dim, per model_version）
```

**Retrain 後 batch user representation 更新（manual_retrain DAG 新增 task）**
```
所有 user 的 item sequence → Model → user_representation table（batch INSERT）
耗時約 1.3 秒（0.06ms × 22,363 users）
```

---



### 2.1 選擇 SID4SRec（SASRec + Diffusion）
**決策：** 使用碩士論文訓練好的 SID4SRec 模型。

**原因：**
- SASRec 是 sequential recommendation 的強 baseline
- Diffusion 增強 + 2 層對比學習提升表現
- 已有訓練好的權重，可直接部署

### 2.2 User representation 雙軌更新策略
**決策：** User representation 存入 DB（`user_representation` table），採雙軌更新：
1. 每次 `GET /recommend` 推論後 UPSERT
2. Retrain 後由 `manual_retrain` DAG 的 `generate_user_representations` task batch 更新所有 user

**原因：**
- RAG `/explain` 需要 user representation 做相似 user 搜尋，必須存 DB
- Batch 更新（~2s，22K users）確保 retrain 後所有 user 的 representation 對齊新 model
- 推論時 UPSERT 確保活躍 user 的 representation 即時更新

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

### 3.4 Blue-Green Deployment（Nginx + Static Config）
**決策：** Nginx 作為 reverse proxy 坐在 serve_blue / serve_green 前面，靜態 config + Docker 內建 DNS 解析 backend，切換時 swap config + `nginx -s reload`。

**架構：**
```
Client → Nginx(:80) → serve_blue  (預設 active)
                    → serve_green (retrain 時啟動，DAG 切換 upstream 後接管)
```

**切換流程（manual_retrain DAG 自動執行）：**
1. 訓練完成後 `docker compose --profile green up -d serve_green`
2. 內網直連 `http://mlops_serve_green:8000/health` 確認 GREEN 正常
3. Swap nginx upstream config（將 `serve_blue:8000` 換成 `serve_green:8000`）
4. `docker exec mlops_nginx nginx -s reload`（reload 是 graceful：worker 會處理完現有連線才退出）
5. `sleep 2 && docker stop mlops_serve_blue`
6. 透過 Nginx 打 `/health` 確認整條路徑正常

**為什麼從 Traefik 換成 Nginx（2026-04-07）：**
- 原本選 Traefik 的理由是「掛 Docker socket → 透過 label 動態感知 backend → 切換不用 reload」。
- 實際在 Docker Desktop on Windows 環境踩到 bug：Traefik 連 `/var/run/docker.sock` 會收到 daemon 回的空 response（`Error response from daemon: ""`），導致 Docker provider 完全讀不到 container labels，所有 router 都註冊不上，外部訪問一律 404。
- 嘗試過的繞法都失敗：
  - 改用 npipe endpoint → Linux 容器不支援 named pipe（"protocol not available"）
  - 加 `tecnativa/docker-socket-proxy` 中介 → proxy 收到 daemon HTTP 400，問題在 Docker Desktop bridge 本身，proxy 擋不住
- Nginx 完全不需要連 Docker daemon，只用 Docker 內建 DNS（`127.0.0.11`）解析 service 名稱，從根本上避開這條失敗路徑。
- 代價：失去「切換不用 reload」的優點，但 `nginx -s reload` 是 graceful 的（在 DAG 裡多一步 `docker exec` 而已），完全可接受。
- 教訓：選技術時不能只看 happy path 的優雅度，要把實際運行環境（Windows + Docker Desktop）的 quirks 算進去。

**目前限制（TODO）：**
- `manual_retrain` DAG 只支援 BLUE→GREEN 單向切換（假設啟動時 active 為 BLUE）
- 下一輪 retrain 要從 GREEN 切回 BLUE 需要偵測當前 active 並反向 swap，目前未實作
- 短期 workaround：retrain 結束後手動把 nginx.conf 改回 `serve_blue:8000`、stop green、start blue

**為什麼不需要 Graceful Draining：**
- 此系統的所有 endpoint（`/recommend`、`/explain` 等）皆為短請求（< 1s）
- 2s `sleep` 提供足夠的 in-flight grace period
- Nginx reload 本身就會 graceful 等舊 worker 處理完連線才結束

**Port 說明：**
- 外部統一透過 Nginx port `80` 存取
- 舊的 `8000`（直連 serve_blue）、`8001`（直連 serve_green）、`8888`（Traefik dashboard）已全部移除

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

### 4.0 推薦參數
| 參數 | 值 |
|------|-----|
| recommend top-k | 20 |

**原因：** 模型 HR@5 = 0.0774，HR@20 = 0.1533。top-20 命中率是 top-5 的兩倍，對推薦品質有顯著改善。

### 4.1 移除 POST /recommend，改為 POST /user

**決策：** 原本的 `POST /recommend`（直接傳 item_sequence 推薦）已移除，改為 `POST /user`。

**原因：**
- `POST /recommend` 傳入的 item_sequence 不會寫進 DB，導致下次 `GET /recommend` 查無歷史（404）
- 實際需求是「新用戶 onboarding」，應先建立 DB 紀錄再推薦，而非繞過 DB
- 系統所有互動資料都在 DB，`GET /recommend` 可以完整處理所有推薦情境

**POST /user 設計：**
- Input：`item_sequence: List[int]`（不傳 user_id，由 DB 自動產生）
- 驗證每個 item_id 存在於 `item` table，否則 404
- `user_id` 用 `nextval('user_id_seq')` 產生（atomic，無競態條件）
- 寫入 `user` + `interaction` table
- 回傳新 `user_id`（22364 為第一個新 user）

### 4.2 Model 在 startup 時載入
**決策：** FastAPI lifespan 啟動時 load model 一次，常駐記憶體。

**原因：** 每次 request 重新 load model 延遲太高（數秒），serving 需要低延遲。

### 4.3 新增 POST /item

**決策：** 新增 `POST /item` endpoint，允許建立新商品。

**設計：**
- Input：`category1: str`、`category2: str（optional）`、`brand: str`、`price: float`
- category / brand 用名稱查詢，不存在則自動建立（UPSERT 邏輯）
- `item_id` 用 `nextval('item_id_seq')` 產生（atomic，無競態條件）
- 回傳新 `item_id`

**原因：** 新用戶 onboarding 時若要帶入系統外的商品，需要先有辦法建立 item，再建立 user。

---

### 4.4 feedback vs interaction 分離
**決策：**
- `POST /feedback`：使用者點擊/購買了推薦的商品
- `POST /interaction`：使用者自行搜尋購買（非推薦來源）

**原因：** 兩者職責不同，feedback 用於評估推薦品質，interaction 用於補充訓練資料。

### 4.5 Implicit feedback only
**決策：** 不設計 explicit feedback（如評分），只記錄行為（點擊/購買）。

**原因：** Amazon Beauty 資料集本身只有 implicit feedback，模型設計也是針對 implicit。

### 4.6 Atomic ID 產生策略（PostgreSQL Sequences）
**決策：** `POST /user`、`POST /item` 使用 PostgreSQL sequence（`nextval()`）產生 ID，而非 `SELECT MAX(id) + 1`。

**原因：**
- `MAX + 1` 是典型競態條件（race condition）：兩個並發請求同時讀到相同 MAX，接著 INSERT 時其中一個會因 PRIMARY KEY 衝突而失敗
- `nextval()` 是 DB-level atomic 操作，天然保證唯一性，不需要額外 lock

**實作細節：**
- `docker/init.sql` 建立四個 sequence：`user_id_seq`、`category_id_seq`、`brand_id_seq`、`item_id_seq`
- 大量注入資料後需執行 `scripts/sync_sequences.sql`，把各 sequence 對齊 `MAX(id)`，避免與已存在的 ID 衝突
- `ingest_beauty.py` 結尾自動執行同步，不需手動呼叫

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

---

## 7. RAG Explanation System

### 7.1 User Representation 儲存策略
**決策：** 每次 `/recommend` 呼叫時 UPSERT `user_representation` table，以 `(user_id, model_version)` 為 key 覆蓋。

**Schema：**
```
user_representation(user_id, model_version, representation vector(192), created_at)
```

**原因：**
- RAG 永遠使用最新 representation，不需要歷史紀錄
- Retrain 後新 model_version 透過 batch task 預先填充，避免 RAG 查無資料

### 7.2 相似 User 查詢參數
| 參數 | 值 |
|------|-----|
| 相似 user top-k | 3 |
| 相似 user cosine similarity threshold | ≥ 0.5 |

### 7.3 LLM 選擇
**決策：** 使用 `models/gemini-2.5-flash` 透過 Google Gemini API。

**原因：**
- Token 量小（百筆以內商品資料），TPM 完全沒問題
- 不需要多模態
- 指令跟隨能力強，輸出乾淨單段純文字（不會 think out loud）
- RAG 整合進 FastAPI（不是獨立服務），架構改動最小

**環境變數：** `GEMINI_API_KEY`、`GEMINI_MODEL=models/gemini-2.5-flash`

**初次嘗試 Gemma-4-31b-it（已棄用）：** 一開始用 `models/gemma-4-31b-it`，但測試發現它會把整個推理過程（Draft 1、Self-Correction、Final Polish 等）全部吐出來，prompt 加再多「不要 think out loud」的指令也壓不下來。Gemma 是 open model 風格，本來就不適合需要乾淨輸出的 production endpoint。換成 Gemini 2.5 Flash 後直接吐出單段純文字，無需後處理。

### 7.4 一般用戶 RAG 流程（兩階段 LLM）
```
輸入：user_id
  ↓
查詢 user_representation + recommendation_log（最新 top-20 item）
  ↓
向量搜尋：找 cosine similarity ≥ 0.5 的前 3 個相似 user
  ↓
取得相似 user 的 item sequence + item 屬性（category, brand）+ recommend_list
  ↓
【第一次 Gemini call】結構化 prompt：
  - 針對每個推薦 item 寫一句話原因
  - 輸出：逐條結構化文字
  ↓
【第二次 Gemini call】摘要 prompt：
  - 把結構化輸出送回 Gemini，產生重點摘要
  ↓
回傳：
{
  "user_id": 123,
  "summary": "一段重點摘要",
  "source": "llm" | "fallback",
  "recommended_items": [...20 個 item 含屬性],
  "user_context": {最近10次互動, top3 categories, top3 brands}
}
```

**兩階段設計原因：**
- top-20 商品直接要 LLM 一段涵蓋品質差（字數爆炸或遺漏）
- 結構化先確保每個 item 有解釋，摘要再整合核心規律
- client 只需要 `summary`，結構化作為中間步驟不回傳

### 7.5 Context 範圍（每個 user）
**決策：** 每個 user（目標 + 相似）的 prompt context 包含：
- 最近 10 次互動的 item（含 category, brand, price，JOIN item table）
- 整段歷史的 top 3 categories（出現次數）
- 整段歷史的 top 3 brands（出現次數）

**原因：**
- 「最近 10 個」抓近期偏好訊號
- 「top 3 cats/brands」抓長期品味訊號（聚合統計，不爆 prompt）
- Prompt size 固定，不會隨 user 歷史長度暴增

### 7.6 新用戶處理（cold start）
**決策：** `/explain` 查不到 `user_representation` 直接回 404，訊息「請先呼叫 POST /user 或 GET /recommend」。

**原因：**
- SASRec 本質：沒互動就沒 representation
- 新用戶應透過 `POST /user`（帶 item_sequence 建立帳號）或 `GET /recommend` 建立歷史
- 不在 `/explain` 內做 fallback inference，避免額外延遲與職責混淆

### 7.7 LLM 失敗處理
**決策：** Gemini API 失敗時回 HTTP 200 + fallback 文字 + `source: "fallback"`，server side 記 log。

**原因：**
- Client 永遠拿到 explanation 字串，UI 一致
- `source` 欄位讓 client 可以區分是真解釋還是 fallback
- 失敗 log 給 server 側追蹤 quota / 網路問題

### 7.8 Code 結構
**決策：** RAG 邏輯獨立成 `rag/` module，serving 只 import。
```
rag/
  context.py    # DB 查詢 → RagContext dataclass
  explain.py    # prompt 組裝 + Gemini API + main entry
serving/main.py # 只掛 GET /explain endpoint
```

**原因：**
- `serving/main.py` 不會肥大
- RAG 內部邏輯可獨立測試（例如 mock Gemini 測 prompt builder）
- 之後加新 RAG endpoint（管理員 dashboard 等）有地方放

### 7.9 多語言支援
**決策：** `/explain?lang=zh|en`，預設 `zh`（繁體中文）。System prompt 模板按 lang 切換，data 段落（item 屬性）保持英文不變。

---

## 8. 分析 Dashboard（Grafana）

### 8.1 工具選擇
**決策：** 使用 Grafana 作為分析 dashboard。

**原因：**
- 直連 PostgreSQL，不需要額外 API
- 標準 SQL 即可完成所有需求
- 加一個 Docker container，架構改動最小

### 8.2 準確率定義
| 狀況 | 定義 | 納入分析 |
|------|------|------|
| **命中** | `/feedback` 互動的 item 在 recommend list 內 | ✅ |
| **未命中** | `/feedback` 互動的 item 不在 recommend list 內 | ✅ |
| `/interaction` 資料 | 直接搜尋行為，非推薦情境 | ❌ |

**準確率 = 命中 / （命中 + 未命中）**

不需要相似度計算，`/feedback` 資料本身已排除直接搜尋行為。

### 8.3 Dashboard Provisioning
**決策：** 使用 Grafana provisioning（YAML + JSON）而非手動 UI 設定。

**原因：**
- Container 重建後自動恢復，不需要重新設定
- Infrastructure as code，版本控制友好

**檔案結構：**
```
grafana/provisioning/
  datasources/postgres.yaml      # PostgreSQL datasource
  dashboards/provider.yaml       # dashboard 載入設定
  dashboards/recommendation_analytics.json  # dashboard 定義
```

**Datasource 密碼管理（2026-04-07）：**
`datasources/postgres.yaml` 的 `user`、`password`、`database` 改用 `$POSTGRES_USER`、`$POSTGRES_PASSWORD`、`$POSTGRES_DB` 插值（Grafana provisioning 原生支援 `$VAR` 從環境變數讀取）。`docker-compose.yml` 的 grafana service 明確傳入這三個環境變數，值沿用 `${POSTGRES_*:-mlops}` 預設值。不再硬寫密碼，換環境部署只改 `.env`。

### 8.4 Dashboard Panels
| Panel | 類型 | 說明 |
|-------|------|------|
| Overall Hit Rate | gauge | 全域命中率，色階 red/yellow/green |
| Total Feedback / Hits / Misses | stat | 累計數字 |
| Hit Rate by Category | bargauge | 各子類別命中率（horizontal，含 ⓘ 說明） |
| Hit Rate by Brand | bargauge | 各品牌命中率（horizontal，含 ⓘ 說明） |
| Hit Rate by User Activity Level | bargauge | 1–10 / 11–20 / >20 互動次數分組，含 ⓘ 說明 |
| Feedback Events Over Time | timeseries | hits / misses 趨勢，GROUP BY hour |

### 8.5 Category Bug Fix
**問題：** `ingest_beauty.py` 原本取 `cats[0]`（第一個非零 category = 頂層 "Beauty"），導致 12,101 個商品的 `category_id1` 全是 "Beauty"，Grafana category drill-down 無意義。

**修法：**
- `ingest_beauty.py` 改為 `cats[-1]`（最後一個非零 = 最具體子類別），與 `data_generator.py` 邏輯對齊
- 新增 `scripts/migrate_item_categories.py`，一次性更新現有 DB 資料（不需重跑 ingestion）
- 執行後 category 分佈正常（Lotions 725 / Nail Polish 712 / Shampoos 494 / ...）

### 8.6 新增 Table Schema
```sql
recommendation_feedback_log(id, user_id, item_id, timestamp, hit)
```
- 由 `POST /feedback` 觸發寫入
- category / brand 查詢時動態 JOIN `item` table
- user 活躍度查詢時動態 JOIN `interaction` table

---

## 9. Testing & CI

### 9.1 整合測試策略
**決策：** 以 FastAPI TestClient + 真實 PostgreSQL fixture 做整合測試為主，不 mock DB。

**原因：**
- 此專案核心邏輯高度依賴 DB schema（FK、sequence、pgvector HNSW index）
- Mock DB 容易讓測試通過但 production 失敗（mock/prod 分歧是常見的 regression 來源）
- `conftest.py` 的 session-scoped fixture 確保每次測試在獨立的測試資料下執行，結束後自動清理

**覆蓋範圍：**
- `tests/test_api.py`：47 個 API 整合測試（42 pass / 3 xfail / 2 LLM skip）
- `tests/test_rag_explain.py`：prompt builder 單元測試（mock Gemini，不需要真實 API key）
- xfail 釘住已知的 validation gap（`/feedback`、`/interaction` 尚未驗證 user/item 存在性）

### 9.2 GitHub Actions CI Pipeline
**決策：** `.github/workflows/ci.yml` 涵蓋 lint + pytest + docker build，在 PR 和 push to main 時觸發。

**原因：**
- 「production-level ML system」沒有 CI 是自我矛盾的
- pytest 整合測試需要起 PostgreSQL service，GitHub Actions 的 service container 原生支援
- Docker build check 確保 image 在 CI 環境可建置，避免「本機可以跑 CI 炸」

**Pipeline 步驟：**
1. `ruff check`（lint）
2. `pytest tests/`（含 PostgreSQL service container，跑完整整合測試）
3. `docker build`（serve image + train image）
