# 🔍 Qdrant Hybrid Search (Dense + Sparse) with Re-ranking

This project demonstrates a **hybrid search engine** using [Qdrant](https://qdrant.tech) that combines **dense** and **sparse** vector search with **binary quantization**, **user-based filtering**, and **late interaction re-ranking** via a cross-encoder model.

It uses a subset of the [Wikipedia 20220301 English dataset](https://huggingface.co/datasets/wikipedia) with randomly assigned user IDs for multi-user simulation.

---

## Environment Setup

1. **Clone the project:**

   ```bash
   git clone https://github.com/your-user/qdrant-hybrid-search.git
   cd qdrant-hybrid-search
   ```

2. **Create and activate a virtual environment (Python 3.11 recommended):**

   ```bash
   python3.11 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## `requirements.txt`

```txt
qdrant-client[fastembed]>=1.8.2
sentence-transformers>=2.2.2
transformers>=4.35.0
fastembed>=0.1.8
datasets>=2.16.0
numpy>=1.24.0
scikit-learn>=1.2.2
tqdm>=4.64.0
```

---

## Project Overview

### `init_qdrant.py`

- Creates a collection named `hybrid_search` with:
  - **Binary Quantization** for dense vectors
  - **Sparse vector config** for Splade
  - Payload indexing on `user_id`
- Loads 1M documents from Wikipedia
- Assigns each document a random `user_id` from 1 to 10
- Ingests in batches of 256 with metadata (`user_id`, `text`)

Run setup:

```bash
python3 init_qdrant.py
```

---

### `search.py`

- Performs **hybrid search** using:
  - Dense: `sentence-transformers/all-MiniLM-L6-v2`
  - Sparse: `prithivida/Splade_PP_en_v1`
- Re-ranks results using:
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`

Run a hybrid search:

```bash
python3 search.py "What is the capital of Uruguay?"
```

With `user_id` filtering:

```bash
python3 search.py "Messi" --user-id 3
```

---

## Output Example

```
Query: What is the capital of Uruguay?

Result #1
ID: 11458
Original Score: 1.0000
Re-rank Score: 0.8942
User ID: 2
Text: Montevideo is the capital and largest city of Uruguay...
————————————————————————————————————————
```

---

## Qdrant Cluster Setup (2 Nodes)

Run the following to start a **2-node Qdrant cluster** using Docker Compose:

### `docker-compose.yml`

```yaml
version: '3.9'

services:
  qdrant_node1:
    image: qdrant/qdrant:latest
    container_name: qdrant_node1
    volumes:
      - qdrant_data1:/qdrant/storage
    ports:
      - "6333:6333"
      - "6334:6334"
      - "6335:6335"
    environment:
      QDRANT__CLUSTER__ENABLED: "true"
      QDRANT__LOG_LEVEL: "INFO"
      QDRANT__SERVICE__HTTP_PORT: "6333"
      QDRANT__SERVICE__GRPC_PORT: "6334"
      QDRANT__CLUSTER__P2P__PORT: "6335"
    command: "./qdrant --uri http://qdrant_node1:6335"
    restart: unless-stopped
    ulimits:
      nofile:
        soft: 65535
        hard: 65535

  qdrant_node2:
    image: qdrant/qdrant:latest
    container_name: qdrant_node2
    volumes:
      - qdrant_data2:/qdrant/storage
    depends_on:
      - qdrant_node1
    ports:
      - "6336:6333"
      - "6337:6334"
      - "6338:6335"
    environment:
      QDRANT__CLUSTER__ENABLED: "true"
      QDRANT__LOG_LEVEL: "INFO"
      QDRANT__SERVICE__HTTP_PORT: "6333"
      QDRANT__SERVICE__GRPC_PORT: "6334"
      QDRANT__CLUSTER__P2P__PORT: "6335"
    command: "./qdrant --bootstrap http://qdrant_node1:6335 --uri http://qdrant_node2:6335"
    restart: unless-stopped
    ulimits:
      nofile:
        soft: 65535
        hard: 65535

volumes:
  qdrant_data1:
    name: qdrant_data1
  qdrant_data2:
    name: qdrant_data2

networks:
  default:
    name: qdrant_network
    driver: bridge
```

### Start the cluster

```bash
docker compose up -d
```

- Qdrant Node 1 accessible at: `http://localhost:6333`
- Qdrant Node 2 accessible at: `http://localhost:6336`

---

## Features Summary

✅ Dense + Sparse hybrid search  
✅ Binary quantization enabled  
✅ Multi-user filtering (`user_id`)  
✅ Late interaction re-ranking  
✅ 2-node Qdrant cluster with replication  
✅ Ingest up to 1M docs (Wikipedia)

---

## Author

**Ignacio Van Droogenbroeck**   
https://github.com/xe-nvdk

---