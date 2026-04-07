# Project Review — MLOps SID4SRec

> 評估日期：2026-04-07（持續更新）

---

## 一句話結論

**這個專案在「系統設計廣度」與「MLOps 完整度」上已達正式履歷專案水準，剩下的問題集中在 runtime 穩健性與程式碼品質。**

目前等級：**8 / 10**

---

## ✅ 已達到「正式履歷專案」水準的部分

| 面向 | 證據 |
|------|------|
| **系統廣度** | 5 個容器化服務（FastAPI / Train / Airflow / Postgres+pgvector / Grafana），完整 MLOps 生命週期 |
| **資料層設計** | `init.sql` schema 完整、有 FK / index / HNSW、model versioning table |
| **設計文件** | `DECISIONS.md`、`README.md` 中英雙語、每個決策都有「為什麼」 |
| **RAG 設計** | `rag/` 模組分離（context.py + explain.py），兩階段 LLM、fallback 處理、cold start 處理 |
| **Pipeline 編排** | 兩條 Airflow DAG，含 backup → train → embeddings → user_repr → Blue-Green 切換 → health_check |
| **API 設計** | 8 endpoints 分 tag、雙語 OpenAPI、dependency injection、Pydantic schema |
| **Observability 入口** | Grafana provisioning（IaC），命中率 / category / brand / 活躍度 drill-down |
| **配置即程式碼** | 完整 Docker Compose、Dockerfile per service、profiles 區分 train/serve/green |
| **Blue-Green 部署** | Nginx + sed swap + `nginx -s reload`，manual_retrain DAG 自動切換，含 pre-flight grep 防呆 |
| **測試** | 47 個 API 整合測試（42 pass / 3 xfail / 2 LLM skip），含 conftest fixture、xfail 釘住已知 gap |
| **CI/CD** | GitHub Actions：lint + pytest + docker build |
| **ID 產生** | `nextval()` sequence 取代 `MAX(id)+1`，無競態條件 |

---

## ❌ 未解決問題（待補強）

### 🔴 重要 1：cold-start `/recommend` 噴 IndexError

**問題：** 對系統中存在但從未被推薦或互動的 item（cold-start item），呼叫 `/recommend` 時會拋出 `IndexError`，而非乾淨的 404 或 fallback。

**原因：** `full_sort_predict` 的 item index 範圍是 `[0, model.item_size)`，但 DB 的 item_id 不一定連續，若 item_id 超出 model index 範圍就會 OOB crash。

**影響：** Demo 時的炸彈——任何只有 cold-start item 的用戶都會噴錯。

**需要做的：** 推論前確認所有 item_id < model.item_size，超出範圍的 item 過濾掉或回 422。

---

### 🟡 重要 2：`/feedback` / `/interaction` 缺 input validation

**問題：** 兩個 endpoint 都沒有驗證 `user_id` 和 `item_id` 是否存在於 DB，直接 INSERT 可能造成 FK 違反。

**影響：** 傳入不存在的 user_id / item_id 會拿到 DB 層的 IntegrityError 而非友善的 404。

**需要做的：** INSERT 前查 `SELECT 1 FROM "user" WHERE user_id = :uid`，不存在就 raise HTTPException 404。

---

### 🟡 重要 3：訓練程式碼用 `print()` 而非結構化 logging

**問題：** `training/trainer.py` 全部用 `print()` 輸出訓練進度，無 log level、無 timestamp、無格式。`config.py` 88 個 hyperparameter 全用 argparse，論文 code 搬過來的氣味重。

**影響：** Docker log 無法接 log aggregator；面試官看 training code 會直接聞到「論文 code 沒整理」的味道。

**需要做的：** 換成 `logging.getLogger(__name__)`，進階可用 Hydra / OmegaConf 管 config。

---

### 🟡 重要 4：Grafana datasource hardcode 密碼

**問題：** `grafana/provisioning/datasources/postgres.yaml` 直接寫死 `password: mlops`，沒走環境變數。

**影響：** secret management 基本門檻沒達到，換環境部署要手動改檔案。

**需要做的：** 改用 Grafana 的 `${GF_DATASOURCE_PASSWORD}` 插值，或走 `secureJsonData`。

---

## 🟢 中等優先（補了會大幅加分）

| # | 項目 | 問題說明 |
|---|------|---------|
| 5 | **README 加截圖** | Grafana dashboard、Swagger UI、Airflow DAG graph — 招募者 30 秒掃 README 決定要不要點進去 |
| 6 | **Gemini API 加 retry + timeout** | 目前失敗一次直接 fallback，`tenacity` decorator 讓暫時性網路問題自動重試 |
| 7 | **Prometheus metrics** | `prometheus-fastapi-instrumentator`，Grafana 目前只看 DB，缺 latency / QPS / error rate |
| 8 | **MLflow 或 W&B 整合** | 訓練實驗追蹤：「我怎麼知道這個 model 比上一版好」 |
| 9 | **資料品質檢查** | ingestion 階段擋掉壞資料（Great Expectations 或自寫 `data_pipeline/validate.py`）|
| 10 | **pyproject.toml + ruff + mypy** | 程式碼品質的最低門檻 |
| 11 | **訓練資料來源閉環** | `DataGenerator` 目前從 raw `.dat` pickle 讀，不依賴 ingestion 結果，pipeline 沒真的閉環 |

---

## 📊 履歷專案成熟度評分

| 面向 | 分數 | 備註 |
|------|------|------|
| 系統設計廣度 | 9 / 10 | 涵蓋面非常廣 |
| 基礎設施 / Docker | 8 / 10 | Nginx Blue-Green 完整；Grafana secret 仍 hardcode |
| 程式碼品質 | 6 / 10 | training/ 有 print()；serving/ 乾淨 |
| 測試 | 7 / 10 | 47 個整合測試；缺 unit/mock 層 |
| CI/CD | 7 / 10 | GitHub Actions 已建；缺 staging deploy check |
| 文件 | 9 / 10 | README + DECISIONS + PROJECT_REVIEW 持續更新 |
| Observability | 6 / 10 | Grafana 有 dashboard；缺 metrics / traces |
| 一致性（設計 vs 實作）| 9 / 10 | 剩 cold-start crash、validation gap |

**總分：8 / 10 — 可以拿去面試，剩下是加分項。**

---

## 附錄：掃描範圍

- `README.md`、`DECISIONS.md`、`CLAUDE.md`
- `docker-compose.yml`、`docker/*.Dockerfile`、`docker/init.sql`、`docker/requirements-*.txt`
- `serving/main.py`
- `rag/context.py`、`rag/explain.py`
- `training/sid4srec.py`、`training/trainer.py`、`training/train.py`、`training/config.py`
- `airflow/dags/manual_retrain.py`、`airflow/dags/monthly_embedding_update.py`
- `scripts/ingest_beauty.py`、`scripts/generate_embeddings.py`、`scripts/generate_user_representations.py`
- `grafana/provisioning/**`
- `.gitignore`、`.env`（脫敏檢查）
