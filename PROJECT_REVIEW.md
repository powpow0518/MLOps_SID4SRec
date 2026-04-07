# Project Review — MLOps SID4SRec

> 履歷專案成熟度評估報告
> 評估日期：2026-04-07
> 評估範圍：整個專案程式碼、infra、文件、設計一致性

---

## 一句話結論

**這個專案在「系統設計廣度」與「MLOps 完整度」上已經接近正式履歷專案，但有幾個會讓資深面試官立刻扣分的硬傷。**

目前等級：**7 / 10 — 看得出有實力，但細節露餡。**
不是小玩具，但還沒到能直接拿去面試 senior 職的成熟度。

---

## ✅ 已經達到「正式履歷專案」水準的部分

| 面向 | 證據 |
|------|------|
| **系統廣度** | 5 個容器化服務（FastAPI / Train / Airflow / Postgres+pgvector / Grafana），完整 MLOps 生命週期 |
| **資料層設計** | `init.sql` schema 完整、有 FK / index / HNSW、model versioning table |
| **設計文件** | `DECISIONS.md` 17KB、`README.md` 中英雙語、每個決策都有「為什麼」 |
| **RAG 設計** | `rag/` 模組分離（context.py + explain.py），兩階段 LLM、fallback 處理、cold start 處理 |
| **Pipeline 編排** | 兩條 Airflow DAG，含 backup → train → embeddings → user_repr → restart → health_check |
| **API 設計** | 8 endpoints 分 tag、雙語 OpenAPI、用 dependency injection、Pydantic schema |
| **Observability 入口** | Grafana provisioning（IaC），命中率 / category / brand / 活躍度 drill-down |
| **配置即程式碼** | 完整 Docker Compose、Dockerfile per service、profiles 區分 train/serve/green |

---

## ❌ 會被資深面試官立刻挑出的硬傷（必須補）

### 🔴 致命 1：完全沒有測試 — ✅ 已修復（2026-04-07）

> `tests/test_api.py` 47 個整合測試（42 pass / 3 xfail / 2 LLM skip），含 conftest 自動清理 fixture、parametrize invalid input、xfail 釘住已知 validation gap。`tests/test_rag_explain.py` 另有 prompt builder 單元測試。



```
$ find tests/ test_*.py *_test.py → 0 個檔案
```

- `serving/main.py` 503 行，0 個 test
- `rag/context.py` 有 SQL 邏輯 + 排序，0 個 test
- ingestion / migration script 0 個 test

**面試官第一句就會問「這怎麼驗證？」**

至少需要：
- `tests/test_serving.py`：FastAPI TestClient + SQLite/Postgres test fixture
- `tests/test_rag_context.py`：mock DB session 測 `find_similar_users`、`get_user_context`
- `tests/test_explain.py`：mock Gemini 驗 prompt builder

---

### 🔴 致命 2：`manual_retrain` DAG 的 run_training 是假的

`airflow/dags/manual_retrain.py:50`

```python
bash_command="echo 'Simulating training... done'",
# bash_command=(  ← 真正指令被註解掉
#     f"docker compose ... train"
# ),
```

**這條 DAG 寫在 README 裡當主打 feature，但實際上 retrain task 不會跑訓練。**
面試官只要打開 dag 看一眼就會發現。

要嘛把真指令打開，要嘛改成 `python -m training.train` 並確保它能 run。

---

### 🔴 致命 3：沒有 CI/CD — ✅ 已修復（2026-04-07）

> `.github/workflows/` 已加入 lint + test + docker build pipeline（commit `3c050d8`）。



- 沒有 `.github/workflows/`
- 沒有 lint / type check / test 自動化
- 沒有 Docker image build check

「production-level ML system」沒有 CI 是矛盾的。
最少要一個 GitHub Actions：lint + pytest + docker build。

---

### 🟡 重要 4：沒有 `.env.example` — ✅ 已修復

README 寫 `cp .env.example .env`，但 repo 裡沒有這個檔案。
Clone 下來的人完全不知道要設哪些變數。

---

### 🟡 重要 5：配置漂移 — ✅ 已修復

> `docker-compose.yml` 預設值已改為 `gemini-2.5-flash`。



`docker-compose.yml:37`

```yaml
GEMINI_MODEL: ${GEMINI_MODEL:-models/gemma-4-31b-it}
```

但 `DECISIONS.md` 7.2.1 明確說 Gemma 已棄用，已換成 `models/gemini-2.5-flash`。
**docker-compose 預設值還停在被棄用的 model。**

---

### 🟡 重要 6：MAX(id)+1 競態條件 — ✅ 已修復

> 全部改用 `nextval('xxx_seq')`，新增 4 個 sequence 到 `init.sql`，並有 `scripts/sync_sequences.sql` 補跑用。



`serving/main.py:352`、`383`、`413`、`422`

```python
'SELECT COALESCE(MAX(user_id), 0) + 1 FROM "user"'
```

**這是教科書級的 race condition。** 並發呼叫 `POST /user` 會撞 ID。
schema 應該用 `SERIAL`，這個一眼就會被資深 backend 點出來。

---

### 🟡 重要 7：Blue-Green 名實不符 — ✅ 已修復（2026-04-07）

> 第一版用 Traefik + container label priority 實作，但在 Docker Desktop on Windows 踩到 socket bridge bug；最終改為 Nginx + 靜態 upstream + sed swap + `nginx -s reload`。`manual_retrain` DAG 已新增 `swap_nginx_upstream` task，含 pre-flight grep 防呆。詳見 `DECISIONS.md` §3.4。目前限制：DAG 只支援 BLUE→GREEN 單向，雙向是 TODO。



`DECISIONS.md` 3.4 自己寫：

> 目前實作：retrain 後直接重啟 serve_blue（因為只更新 model weights，不換 code），Blue-Green 保留供未來 code 更新時使用。

**README 主打「Blue-Green deployment」但實際上沒做。**
面試官追問「怎麼切流量」時會穿幫。
要嘛實作（哪怕是 nginx 路由切換），要嘛從 README 拿掉。

---

### 🟡 重要 8：訓練程式碼的維護度 — ⚠️ 部分修復

> `sid4srec.py` 130 行 commented-out code 已清除、`trainer.py` typo 已修。logging / Hydra / argparse 仍未處理。



- `training/sid4srec.py:412-542`：130 行 commented-out code 沒清掉
- `training/trainer.py:17`：`# define the start epoch for keepon trainingzhonss` ← typo
- 大量 `print()` 而非 structured logging
- `config.py` 88 個 hyperparameter 全用 argparse（履歷專案常用 Hydra/OmegaConf）

這些不是 bug，但「論文 code 直接搬過來沒整理」的氣味很重。

---

## 🟢 中等優先（補了會大幅加分）

| # | 項目 | 為什麼重要 |
|---|------|-----------|
| 9 | **README 加截圖**：Grafana dashboard、Swagger UI、Airflow DAG graph | 招募者 30 秒掃 README 決定要不要點進去 |
| 10 | **Gemini API 加 retry + timeout**：`tenacity` decorator | 目前失敗一次直接 fallback，production 不夠 robust |
| 11 | **structured logging**：`logging.config` + JSON formatter | 取代 `print()` |
| 12 | **加 Prometheus metrics**：`prometheus-fastapi-instrumentator` | Grafana 只看 DB 不夠，要有 latency / QPS / error rate |
| 13 | **MLflow 或 W&B 整合**：訓練實驗追蹤 | 「我怎麼知道這個 model 比上一版好」 |
| 14 | **資料品質檢查**：Great Expectations 或自寫 `data_pipeline/validate.py` | ingestion 階段擋掉壞資料 |
| 15 | **pyproject.toml + ruff + mypy** | 程式碼品質的最低門檻 |
| 16 | **Grafana datasource 用 env var**：不要 hardcode `password: mlops` | secret management 基本動作 |
| 17 | **訓練資料來源閉環**：`DataGenerator` 從 DB 讀，不是 raw `.dat` pickle | 目前訓練不依賴 ingestion 結果，pipeline 沒真的閉環 |

---

## 📊 履歷專案成熟度評分

| 面向 | 原始分 | 更新分（2026-04-07） | 備註 |
|------|-------|---------------------|------|
| 系統設計廣度 | 9 / 10 | 9 / 10 | 涵蓋面非常廣 |
| 基礎設施 / Docker | 8 / 10 | 8 / 10 | Nginx 取代 Traefik 後更穩；secret 漏洞仍在 |
| 程式碼品質 | 5 / 10 | 6 / 10 | training/ 已局部清理 |
| **測試** | **0 / 10** | **7 / 10** | 47 個整合測試 + 3 xfail 釘 gap，缺 unit/mock 層 |
| **CI/CD** | **0 / 10** | **7 / 10** | GitHub Actions（lint + test + docker build）已建 |
| 文件 | 9 / 10 | 9 / 10 | README + DECISIONS + PROJECT_REVIEW 持續更新 |
| Observability | 6 / 10 | 6 / 10 | 仍缺 metrics / traces |
| 一致性（設計 vs 實作） | 5 / 10 | 8 / 10 | Blue-Green 真的做了、預設值對齊、race condition 修了；剩 DAG 假 retrain |

**原始總分：6.5 / 10 → 更新後：7.5 / 10**

剩下扣分集中在：cold-start IndexError、`/feedback`/`/interaction` 缺 validation、manual_retrain 的 run_training 仍是 echo、Grafana datasource 還在 hardcode 密碼。

---

## 🎯 在「最後一步」之前的補強優先序

如果只想做最少的事讓這個專案脫離「玩具」標籤、進入「正式履歷」階段，照這個順序：

| 順序 | 任務 | 狀態 |
|------|------|------|
| 1 | 打開 `manual_retrain` 的真實訓練指令 → 不能讓 demo 是假的 | 待處理 |
| 2 | 新增 `.env.example` → 別人 clone 才能跑 | ✅ 已完成 |
| 3 | 修 `docker-compose.yml` 的 `GEMINI_MODEL` 預設值 → 一致性 | ✅ 已完成 |
| 4 | 用 sequence `nextval` 取代 `MAX(id)+1` → race condition | ✅ 已完成（`init.sql` + `serving/main.py` + `ingest_beauty.py` + `scripts/sync_sequences.sql`）|
| 5 | 加 `tests/` 目錄，至少 5–10 個 pytest → 砍掉「0 / 10」這個尷尬數字 | 待處理 |
| 6 | 加 `.github/workflows/ci.yml`：lint + pytest + docker build | 待處理 |
| 7 | README 補 3 張截圖（Grafana / Swagger / Airflow） | 待處理 |
| 8 | 清掉 `sid4srec.py` 的 130 行註解 code + typo | ✅ 已完成 |

**做完 1–8，這個專案會從 6.5 升到 8.5，可以掛在 GitHub 上面試用。**

---

## 附錄：掃描範圍

本評估覆蓋的檔案：

- `README.md`、`DECISIONS.md`、`CLAUDE.md`
- `docker-compose.yml`、`docker/*.Dockerfile`、`docker/init.sql`、`docker/requirements-*.txt`
- `serving/main.py`
- `rag/context.py`、`rag/explain.py`
- `training/sid4srec.py`、`training/trainer.py`、`training/train.py`、`training/config.py`
- `airflow/dags/manual_retrain.py`、`airflow/dags/monthly_embedding_update.py`
- `scripts/ingest_beauty.py`、`scripts/generate_embeddings.py`、`scripts/generate_user_representations.py`
- `grafana/provisioning/**`
- `.gitignore`、`.env`（脫敏檢查）
