# MLOps_SID4SRec

[English](#english) | [繁體中文](#繁體中文)

---

<a name="english"></a>

## English

A production-level ML system for sequential recommendation, built as a hands-on MLOps learning project.

### Overview

This system serves personalized item recommendations using **SID4SRec** — a SASRec-based model enhanced with diffusion augmentation and contrastive learning, originally developed in a master's thesis. The project is designed to cover the full MLOps lifecycle: data ingestion, model training, serving, and automated retraining pipelines.

**Dataset:** Amazon Beauty (22,363 users / 12,101 items / 198,502 interactions)

---

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Docker Compose                    │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │ serve    │   │ train    │   │ airflow        │  │
│  │ (port    │   │ (profile │   │ (scheduler +   │  │
│  │  8000)   │   │  train)  │   │  webserver)    │  │
│  └────┬─────┘   └────┬─────┘   └───────┬────────┘  │
│       │              │                 │            │
│       └──────────────┴────────┬────────┘            │
│                               │                     │
│                    ┌──────────▼──────────┐          │
│                    │  PostgreSQL +        │          │
│                    │  pgvector (pg16)     │          │
│                    └─────────────────────┘          │
└─────────────────────────────────────────────────────┘
```

- **Serving**: FastAPI, model loaded at startup, Blue-Green deployment ready
- **Training**: PyTorch 2.0 + CUDA 11.7, triggered via Airflow or manually
- **Orchestration**: Apache Airflow with two DAGs
- **Storage**: PostgreSQL for interactions/metadata + pgvector for item embeddings (192-dim)

---

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | SID4SRec (SASRec + Diffusion + Contrastive Learning) |
| Serving | FastAPI |
| Orchestration | Apache Airflow |
| Database | PostgreSQL 16 + pgvector |
| Infrastructure | Docker / Docker Compose |
| Language | Python |

---

### Project Structure

```
.
├── serving/          # FastAPI application
├── training/         # SID4SRec model code (sid4srec.py, trainer.py, ...)
├── airflow/dags/     # Airflow DAGs (manual_retrain, monthly_embedding_update)
├── data_pipeline/    # Data ingestion utilities
├── scripts/          # One-off scripts (ingest, generate embeddings, ...)
├── docker/           # Dockerfiles + requirements per service
├── models/           # Model weights (not committed)
├── data/             # Raw and processed data (not committed)
├── docker-compose.yml
└── DECISIONS.md      # Full design decision log
```

---

### Key Design Decisions

- **Single PostgreSQL instance** for both relational data and vector search (pgvector HNSW) — dataset scale doesn't warrant a dedicated vector DB
- **User embedding computed at inference time** — SASRec user representation depends on item sequence, not a static vector
- **Train/Serve container separation** — train service uses `profiles: [train]` to avoid accidental startup; triggered by Airflow via `docker compose run`
- **Blue-Green deployment** — serve_blue (port 8000) always on; serve_green (port 8001) used during model updates
- **Model naming convention** — always `best_model.pt`; retrain backs up to `best_model_prev.pt` for rollback

See [DECISIONS.md](./DECISIONS.md) for full rationale on every design choice.

---

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/recommend` | Recommend based on DB interaction history |
| POST | `/recommend` | Recommend based on provided item sequence |
| POST | `/feedback` | Record click/purchase of a recommended item |
| POST | `/interaction` | Record organic interaction (non-recommendation) |

---

### Airflow DAGs

| DAG | Trigger | Description |
|-----|---------|-------------|
| `manual_retrain` | Manual | Full retrain + restart serving |
| `monthly_embedding_update` | Cron (1st of month, 02:00) | Regenerate item embeddings with current model |

---

### Getting Started

```bash
# Copy and configure environment
cp .env.example .env  # edit MLOPS_PROJECT_DIR to your absolute path

# Start all services (except train)
docker compose up -d

# Run data ingestion
docker compose run --rm train python -m scripts.ingest_beauty

# Generate item embeddings
docker compose run --rm train python -m scripts.generate_embeddings

# Trigger retraining manually
docker compose --profile train run --no-deps --rm train python -m training.train
```

---

### Status

This is an active learning project. Current progress:

- [x] DB schema design (PostgreSQL + pgvector)
- [x] Docker infrastructure (train/serve/airflow separation)
- [x] SID4SRec model integration
- [x] FastAPI serving layer
- [x] Airflow DAG skeleton
- [ ] RAG-based explanation system
- [ ] Full end-to-end pipeline validation

---

---

<a name="繁體中文"></a>

## 繁體中文

以序列推薦為核心的 MLOps 實作學習專案，目標是建立一套具備生產水準的 ML 系統。

### 專案概述

本系統使用 **SID4SRec** 模型提供個人化商品推薦。SID4SRec 以 SASRec 為基礎，結合擴散模型（Diffusion）增強與對比學習，源自碩士論文研究成果。專案涵蓋完整的 MLOps 生命週期：資料注入、模型訓練、模型服務、自動化重訓 Pipeline。

**資料集：** Amazon Beauty（22,363 位使用者 / 12,101 件商品 / 198,502 筆互動紀錄）

---

### 系統架構

```
┌─────────────────────────────────────────────────────┐
│                   Docker Compose                    │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │ serve    │   │ train    │   │ airflow        │  │
│  │ (port    │   │ (profile │   │ (scheduler +   │  │
│  │  8000)   │   │  train)  │   │  webserver)    │  │
│  └────┬─────┘   └────┬─────┘   └───────┬────────┘  │
│       │              │                 │            │
│       └──────────────┴────────┬────────┘            │
│                               │                     │
│                    ┌──────────▼──────────┐          │
│                    │  PostgreSQL +        │          │
│                    │  pgvector (pg16)     │          │
│                    └─────────────────────┘          │
└─────────────────────────────────────────────────────┘
```

- **Serving**：FastAPI，啟動時載入模型常駐記憶體，支援 Blue-Green Deployment
- **Training**：PyTorch 2.0 + CUDA 11.7，由 Airflow 觸發或手動執行
- **Orchestration**：Apache Airflow，管理兩條 DAG
- **Storage**：PostgreSQL 存放互動紀錄與商品中繼資料；pgvector 存放 192-dim item embedding

---

### 技術棧

| 元件 | 技術 |
|------|------|
| 模型 | SID4SRec（SASRec + Diffusion + 對比學習） |
| 服務層 | FastAPI |
| 排程 | Apache Airflow |
| 資料庫 | PostgreSQL 16 + pgvector |
| 基礎設施 | Docker / Docker Compose |
| 語言 | Python |

---

### 專案結構

```
.
├── serving/          # FastAPI 應用程式
├── training/         # SID4SRec 模型程式碼（sid4srec.py, trainer.py, ...）
├── airflow/dags/     # Airflow DAG（manual_retrain, monthly_embedding_update）
├── data_pipeline/    # 資料注入工具
├── scripts/          # 一次性腳本（資料注入、產生 embedding 等）
├── docker/           # 各服務的 Dockerfile 與 requirements
├── models/           # 模型權重（不納入版本控制）
├── data/             # 原始與處理後資料（不納入版本控制）
├── docker-compose.yml
└── DECISIONS.md      # 完整設計決策紀錄
```

---

### 主要設計決策

- **單一 PostgreSQL 實例**同時處理關聯式資料與向量搜尋（pgvector HNSW）— 資料規模不需要獨立向量資料庫
- **User Embedding 在推論時即時計算** — SASRec 的 user representation 依賴 item sequence，非靜態向量，存 DB 無意義
- **Train / Serve container 分離** — train service 使用 `profiles: [train]`，避免意外啟動；由 Airflow 透過 `docker compose run` 觸發
- **Blue-Green Deployment** — serve_blue（port 8000）常駐；serve_green（port 8001）在模型更新時啟動驗證
- **模型命名慣例** — 統一使用 `best_model.pt`；重訓前備份為 `best_model_prev.pt` 供 rollback

完整設計決策與原因請見 [DECISIONS.md](./DECISIONS.md)。

---

### API 端點

| Method | 路徑 | 說明 |
|--------|------|------|
| GET | `/recommend` | 從 DB 查詢歷史互動紀錄後推薦 |
| POST | `/recommend` | 直接接收 item sequence 後推薦 |
| POST | `/feedback` | 記錄使用者點擊 / 購買推薦商品 |
| POST | `/interaction` | 記錄使用者自行搜尋購買（非推薦來源） |

---

### Airflow DAG

| DAG | 觸發方式 | 說明 |
|-----|---------|------|
| `manual_retrain` | 手動 | 完整重訓 + 重啟 serving |
| `monthly_embedding_update` | Cron（每月 1 日 02:00） | 以現有模型重新產生 item embedding |

---

### 快速開始

```bash
# 複製並設定環境變數
cp .env.example .env  # 修改 MLOPS_PROJECT_DIR 為你的絕對路徑

# 啟動所有服務（不含 train）
docker compose up -d

# 執行資料注入
docker compose run --rm train python -m scripts.ingest_beauty

# 產生 item embedding
docker compose run --rm train python -m scripts.generate_embeddings

# 手動觸發重訓
docker compose --profile train run --no-deps --rm train python -m training.train
```

---

### 目前進度

這是一個持續進行中的學習專案，目前完成狀態：

- [x] DB Schema 設計（PostgreSQL + pgvector）
- [x] Docker 基礎設施（train / serve / airflow 分離）
- [x] SID4SRec 模型整合
- [x] FastAPI Serving 層
- [x] Airflow DAG 骨架
- [ ] RAG-based 推薦解釋系統
- [ ] 完整端到端 Pipeline 驗證
