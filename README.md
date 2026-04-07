# MLOps_SID4SRec

[English](#english) | [繁體中文](#繁體中文)

---

<a name="english"></a>

## English

A production-level ML system for sequential recommendation, built as a hands-on MLOps learning project.

### Overview

This system serves personalized item recommendations using **SID4SRec** — a SASRec-based model enhanced with diffusion augmentation and contrastive learning, originally developed in a master's thesis. The project is designed to cover the full MLOps lifecycle: data ingestion, model training, serving, automated retraining pipelines, RAG-based explanation, and analytics dashboards.

**Dataset:** Amazon Beauty (22,363 users / 12,101 items / 198,502 interactions)

---

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Docker Compose                         │
│                                                              │
│  Client → Traefik(:80) ──priority=100──▶ serve_green        │
│                      └──priority=1─────▶ serve_blue         │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐      │
│  │ traefik  │  │ train    │  │ airflow  │  │grafana │      │
│  │ (port 80)│  │ (profile │  │(sched +  │  │(port   │      │
│  │ dash:8888│  │  train)  │  │ websvr)  │  │ 3000)  │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘      │
│       │             │             │             │           │
│       └─────────────┴──────┬──────┘             │           │
│                            │        ┌───────────┘           │
│                 ┌──────────▼────────▼──────┐                │
│                 │  PostgreSQL + pgvector    │                │
│                 │  (pg16, port 5432)        │                │
│                 └───────────────────────────┘                │
└──────────────────────────────────────────────────────────────┘
```

- **Serving**: FastAPI, model loaded at startup, Blue-Green deployment via Traefik router priority
- **Training**: PyTorch 2.0 + CUDA 11.7, triggered via Airflow or manually
- **Orchestration**: Apache Airflow with two DAGs
- **Storage**: PostgreSQL for interactions/metadata + pgvector for item embeddings (192-dim) and user representations (192-dim)
- **Analytics**: Grafana (port 3000), direct PostgreSQL connection, recommendation accuracy dashboards

---

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | SID4SRec (SASRec + Diffusion + Contrastive Learning) |
| Serving | FastAPI |
| Orchestration | Apache Airflow |
| Database | PostgreSQL 16 + pgvector |
| Analytics | Grafana |
| Infrastructure | Docker / Docker Compose |
| Language | Python |

---

### Project Structure

```
.
├── serving/          # FastAPI application + RAG explanation (rag/)
├── training/         # SID4SRec model code (sid4srec.py, trainer.py, ...)
├── airflow/dags/     # Airflow DAGs (manual_retrain, monthly_embedding_update)
├── data_pipeline/    # Data ingestion utilities
├── scripts/          # One-off scripts (ingest, generate embeddings, generate user representations, ...)
├── docker/           # Dockerfiles + requirements per service
├── grafana/          # Grafana provisioning (datasource + dashboard JSON)
├── models/           # Model weights (not committed)
├── data/             # Raw and processed data (not committed)
├── docker-compose.yml
└── DECISIONS.md      # Full design decision log
```

---

### Key Design Decisions

- **Single PostgreSQL instance** for both relational data and vector search (pgvector HNSW) — dataset scale doesn't warrant a dedicated vector DB
- **User representation**: computed at inference time (UPSERT on every `/recommend`) + batch-regenerated for all users after each retrain (~2s for 22K users, batch=256)
- **Train/Serve container separation** — train service uses `profiles: [train]` to avoid accidental startup; triggered by Airflow via `docker compose run`
- **Blue-Green deployment** — serve_blue (port 8000) always on; serve_green (port 8001) used during model updates
- **Model naming convention** — always `best_model.pt`; retrain backs up to `best_model_prev.pt` for rollback
- **RAG explanation** — two-step Gemini API (structured per-item → summary paragraph); integrated into FastAPI; finds top-3 similar users (cosine ≥ 0.5) via pgvector; returns `summary`, `recommended_items` (top-20), and `user_context`
- **Recommend top-k = 20** — HR@5=0.0774, HR@20=0.1533; top-20 doubles hit rate

See [DECISIONS.md](./DECISIONS.md) for full rationale on every design choice.

---

### API Endpoints

| Method | Path | Tag | Description |
|--------|------|-----|-------------|
| GET | `/recommend` | recommendation | Recommend based on DB interaction history |
| GET | `/user_list` | recommendation | List all user IDs |
| POST | `/feedback` | interaction | Record click/purchase of a recommended item |
| POST | `/interaction` | interaction | Record organic interaction (non-recommendation) |
| GET | `/explain` | RAG_explanation | RAG-based natural language explanation (`?user_id=X&lang=zh\|en`) |
| POST | `/user` | user_management | Register new user with initial item sequence (auto-generates user_id) |
| POST | `/item` | user_management | Create new item with category, brand, and price |
| GET | `/health` | system | Health check |

---

### Airflow DAGs

| DAG | Trigger | Description |
|-----|---------|-------------|
| `manual_retrain` | Manual | backup → train → generate_embeddings → generate_user_representations → restart_serving → health_check |
| `monthly_embedding_update` | Cron (1st of month, 02:00) | Regenerate item embeddings with current model |

---

### Getting Started

```bash
# Copy and configure environment
cp .env.example .env  # edit MLOPS_PROJECT_DIR and GEMINI_API_KEY

# Start all services (except train)
docker compose up -d

# Run data ingestion
docker compose run --rm train python -m scripts.ingest_beauty

# Sync DB sequences (required after ingestion — ensures POST /user and POST /item
# generate non-conflicting IDs via nextval rather than MAX+1)
docker compose exec postgres psql -U mlops -d mlops -f /dev/stdin < scripts/sync_sequences.sql

# Generate item embeddings
docker compose run --rm train python -m scripts.generate_embeddings

# Generate user representations (all users)
docker compose run --rm train python -m scripts.generate_user_representations

# Trigger retraining manually
docker compose --profile train run --no-deps --rm train python -m training.train
```

---

### Status

This is an active learning project. Current progress:

- [x] DB schema design (PostgreSQL + pgvector)
- [x] Docker infrastructure (train/serve/airflow/grafana separation)
- [x] SID4SRec model integration
- [x] FastAPI serving layer (8 endpoints)
- [x] Airflow DAG skeleton
- [x] User representation pipeline (inference-time UPSERT + post-retrain batch)
- [x] RAG-based explanation system (`/explain`, Google Gemini API)
- [x] Grafana analytics dashboard (recommendation accuracy, drill-down by category/brand/user activity, provisioning)
- [ ] Full end-to-end pipeline validation

---

---

<a name="繁體中文"></a>

## 繁體中文

以序列推薦為核心的 MLOps 實作學習專案，目標是建立一套具備生產水準的 ML 系統。

### 專案概述

本系統使用 **SID4SRec** 模型提供個人化商品推薦。SID4SRec 以 SASRec 為基礎，結合擴散模型（Diffusion）增強與對比學習，源自碩士論文研究成果。專案涵蓋完整的 MLOps 生命週期：資料注入、模型訓練、模型服務、自動化重訓 Pipeline、RAG 推薦解釋，以及分析儀表板。

**資料集：** Amazon Beauty（22,363 位使用者 / 12,101 件商品 / 198,502 筆互動紀錄）

---

### 系統架構

```
┌──────────────────────────────────────────────────────────────┐
│                       Docker Compose                         │
│                                                              │
│  Client → Traefik(:80) ──priority=100──▶ serve_green        │
│                      └──priority=1─────▶ serve_blue         │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐      │
│  │ traefik  │  │ train    │  │ airflow  │  │grafana │      │
│  │ (port 80)│  │ (profile │  │(排程 +   │  │(port   │      │
│  │ dash:8888│  │  train)  │  │ websvr)  │  │ 3000)  │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘      │
│       │             │             │             │           │
│       └─────────────┴──────┬──────┘             │           │
│                            │        ┌───────────┘           │
│                 ┌──────────▼────────▼──────┐                │
│                 │  PostgreSQL + pgvector    │                │
│                 │  (pg16, port 5432)        │                │
│                 └───────────────────────────┘                │
└──────────────────────────────────────────────────────────────┘
```

- **Serving**：FastAPI，啟動時載入模型常駐記憶體；Traefik 透過 container label priority 實現 Blue-Green 零 downtime 切換
- **Training**：PyTorch 2.0 + CUDA 11.7，由 Airflow 觸發或手動執行
- **Orchestration**：Apache Airflow，管理兩條 DAG
- **Storage**：PostgreSQL 存放互動紀錄與商品中繼資料；pgvector 存放 192-dim item embedding 與 user representation
- **Analytics**：Grafana（port 3000），直連 PostgreSQL，提供推薦準確率分析儀表板

---

### 技術棧

| 元件 | 技術 |
|------|------|
| 模型 | SID4SRec（SASRec + Diffusion + 對比學習） |
| 服務層 | FastAPI |
| 排程 | Apache Airflow |
| 資料庫 | PostgreSQL 16 + pgvector |
| 分析 | Grafana |
| 基礎設施 | Docker / Docker Compose |
| 語言 | Python |

---

### 專案結構

```
.
├── serving/          # FastAPI 應用程式 + RAG 解釋模組（rag/）
├── training/         # SID4SRec 模型程式碼（sid4srec.py, trainer.py, ...）
├── airflow/dags/     # Airflow DAG（manual_retrain, monthly_embedding_update）
├── data_pipeline/    # 資料注入工具
├── scripts/          # 一次性腳本（資料注入、產生 embedding、產生 user representation 等）
├── docker/           # 各服務的 Dockerfile 與 requirements
├── grafana/          # Grafana provisioning（datasource + dashboard JSON）
├── models/           # 模型權重（不納入版本控制）
├── data/             # 原始與處理後資料（不納入版本控制）
├── docker-compose.yml
└── DECISIONS.md      # 完整設計決策紀錄
```

---

### 主要設計決策

- **單一 PostgreSQL 實例**同時處理關聯式資料與向量搜尋（pgvector HNSW）— 資料規模不需要獨立向量資料庫
- **User Representation 雙軌更新**：每次 `/recommend` 推論時即時 UPSERT；重訓後由 DAG batch 更新全部 user（~2s，22K users，batch=256）
- **Train / Serve container 分離** — train service 使用 `profiles: [train]`，避免意外啟動；由 Airflow 透過 `docker compose run` 觸發
- **Blue-Green Deployment** — serve_blue（port 8000）常駐；serve_green（port 8001）在模型更新時啟動驗證
- **模型命名慣例** — 統一使用 `best_model.pt`；重訓前備份為 `best_model_prev.pt` 供 rollback
- **RAG 解釋系統** — 兩階段 Gemini API（逐條結構化 → 重點摘要）；整合進 FastAPI；透過 pgvector 找 top-3 相似 user（cosine ≥ 0.5）；回傳 `summary`、`recommended_items`（top-20）、`user_context`
- **Recommend top-k = 20** — HR@5=0.0774，HR@20=0.1533；top-20 命中率是 top-5 的兩倍

完整設計決策與原因請見 [DECISIONS.md](./DECISIONS.md)。

---

### API 端點

| Method | 路徑 | 分組 | 說明 |
|--------|------|------|------|
| GET | `/recommend` | recommendation | 從 DB 查詢歷史互動紀錄後推薦 |
| GET | `/user_list` | recommendation | 列出所有 user ID |
| POST | `/feedback` | interaction | 記錄使用者點擊 / 購買推薦商品 |
| POST | `/interaction` | interaction | 記錄使用者自行搜尋購買（非推薦來源） |
| GET | `/explain` | RAG_explanation | RAG 自然語言推薦解釋（`?user_id=X&lang=zh\|en`） |
| POST | `/user` | user_management | 建立新用戶並寫入初始 item sequence（user_id 自動產生）|
| POST | `/item` | user_management | 建立新商品（含 category、brand、price）|
| GET | `/health` | system | 健康檢查 |

---

### Airflow DAG

| DAG | 觸發方式 | 說明 |
|-----|---------|------|
| `manual_retrain` | 手動 | backup → 重訓 → generate_embeddings → generate_user_representations → 重啟 serving → health_check |
| `monthly_embedding_update` | Cron（每月 1 日 02:00） | 以現有模型重新產生 item embedding |

---

### 快速開始

```bash
# 複製並設定環境變數
cp .env.example .env  # 修改 MLOPS_PROJECT_DIR 與 GEMINI_API_KEY

# 啟動所有服務（不含 train）
docker compose up -d

# 執行資料注入
docker compose run --rm train python -m scripts.ingest_beauty

# 同步 DB sequence（注入後必須執行一次——確保 POST /user、POST /item 透過 nextval 產生不衝突的 ID）
docker compose exec postgres psql -U mlops -d mlops -f /dev/stdin < scripts/sync_sequences.sql

# 產生 item embedding
docker compose run --rm train python -m scripts.generate_embeddings

# 產生所有 user representation
docker compose run --rm train python -m scripts.generate_user_representations

# 手動觸發重訓
docker compose --profile train run --no-deps --rm train python -m training.train
```

---

### 目前進度

這是一個持續進行中的學習專案，目前完成狀態：

- [x] DB Schema 設計（PostgreSQL + pgvector）
- [x] Docker 基礎設施（train / serve / airflow / grafana 分離）
- [x] SID4SRec 模型整合
- [x] FastAPI Serving 層（8 個 endpoint）
- [x] Airflow DAG 骨架
- [x] User Representation Pipeline（推論時 UPSERT + 重訓後 batch 更新）
- [x] RAG-based 推薦解釋系統（`/explain`，Google Gemini API）
- [x] Grafana 分析儀表板（推薦準確率，依 category / brand / 用戶活躍度 drill-down，provisioning 自動載入）
- [ ] 完整端到端 Pipeline 驗證
