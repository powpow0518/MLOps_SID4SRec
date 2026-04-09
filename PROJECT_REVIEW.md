# Project Review — MLOps SID4SRec

> 評估日期：2026-04-07（持續更新）

---

## 一句話結論

**這個專案在「系統設計廣度」與「MLOps 完整度」上已達正式履歷專案水準，剩下的問題集中在 runtime 穩健性與程式碼品質。**

目前等級：**8.5 / 10**

---

## ✅ 已達到「正式履歷專案」水準的部分

| 面向 | 證據 |
|------|------|
| **系統廣度** | 5 個容器化服務（FastAPI / Train / Airflow / Postgres+pgvector / Grafana），完整 MLOps 生命週期 |
| **資料層設計** | `init.sql` schema 完整、有 FK / index / HNSW、model versioning table |
| **設計文件** | `DECISIONS.md`、`README.md` 中英雙語、每個決策都有「為什麼」 |
| **RAG 設計** | `rag/` 模組分離（context.py + explain.py），兩階段 LLM、fallback 處理、cold start 處理 |
| **Pipeline 編排** | 兩條 Airflow DAG，含 snapshot → backup → train → embeddings → user_repr → Blue-Green 切換 → health_check → auto rollback |
| **API 設計** | 8 endpoints 分 tag、雙語 OpenAPI、dependency injection、Pydantic schema |
| **Observability 入口** | Grafana provisioning（IaC），命中率 / category / brand / 活躍度 drill-down |
| **配置即程式碼** | 完整 Docker Compose、Dockerfile per service、profiles 區分 train/serve/green |
| **Blue-Green 部署** | Nginx + sed swap + `nginx -s reload`，manual_retrain DAG 自動切換，含 pre-flight grep 防呆 |
| **測試** | 47 個 API 整合測試（46 pass / 2 LLM skip），含 conftest fixture、serving/dependencies 覆蓋 input validation |
| **CI/CD** | GitHub Actions：lint + pytest + docker build |
| **ID 產生** | `nextval()` sequence 取代 `MAX(id)+1`，無競態條件 |

---

---

## 📊 履歷專案成熟度評分

| 面向 | 分數 | 備註 |
|------|------|------|
| 系統設計廣度 | 9 / 10 | 涵蓋面非常廣 |
| 基礎設施 / Docker | 9 / 10 | Nginx Blue-Green 完整；Grafana datasource 走環境變數 |
| 程式碼品質 | 7.5 / 10 | training/ 已改 logging；ruff 規則擴到 I/B/UP/SIM；缺 mypy |
| 測試 | 7 / 10 | 47 個整合測試；input validation 覆蓋；缺 unit/mock 層 |
| CI/CD | 7 / 10 | GitHub Actions 已建；缺 staging deploy check |
| 文件 | 9.5 / 10 | README + DECISIONS + PROJECT_REVIEW 持續更新；README 加入截圖 |
| Observability | 6 / 10 | Grafana 有 dashboard；缺 metrics / traces |
| 一致性（設計 vs 實作）| 10 / 10 | cold-start 已修；input validation 已補；train/DB vocab remap 已修 |

**總分：8.5 / 10 — 可以拿去面試，剩下是加分項。**

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
