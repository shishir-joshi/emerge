# Training Embedding Models on your domain with `Emerge`

Curate high-quality, relevance-weighted, topic-tagged datasets from your raw text — to train and fine-tune embedding models that actually understand your domain.

## The problem:

Training embedding models on domain-specific data is challenging due to the lack of high-quality, labeled datasets. Traditional methods often rely on generic datasets that do not capture the nuances of specific domains, leading to suboptimal model performance.
Can we generate structured datasets from unstructured text, with minimal human effort?

## What is Emerge?

Emerge discovers, extracts, and organizes semantically meaningful spans from unstructured text, producing a structured dataset (spans + cluster/topic labels + relevance/quality scores). This dataset is ready-made for:
- Fine-tuning existing embedding models on your data (e.g., GTE, BGE, E5, Jina)
- Training new domain-specific embedders from scratch
- Evaluating embedding quality for search, RAG, clustering, and deduplication

Two paths to fit your custom data:
1. **Unsupervised discovery** (`LocalClusterAnalyzer`) – discover emergent topics and mine positive/negative pairs from clusters.  
2. **Topic-guided** (`LateInteractionCrossEncoder`) – use your topic definitions (name/description/examples) to produce labeled spans per topic.

High-level pipeline:
1. Dense, overlapping span generation  
2. Embedding  
3. UMAP + HDBSCAN clustering (unsupervised) **or** late-interaction scoring (topic-guided)  
4. Pareto selection + relevance/quality metrics  
5. Structured dataset export (CSV/Parquet) for embedding training  

---

## Minimal Emerge flow (reference)

```python
# Unsupervised discovery
from emerge.src.curation.curator import Curator, CuratorConfig
from emerge.src.curation.analyzers.local_cluster_analyzer import LocalClusterAnalyzer
from emerge.src.embedding.clustering import ClusteringPipeline
from emerge.src.curation.analyzers.base_analyzer import AnalyzerConfig

pipeline = ClusteringPipeline.load("artifacts/clustering/models")
analyzer = LocalClusterAnalyzer(
    pipeline=pipeline,
    config=AnalyzerConfig(min_len=5, max_len=50),
    model=model,
    tokenizer=tokenizer,
)

config  = CuratorConfig(min_length=10, max_length=200, quality_threshold=0.6)
curator = Curator(model, tokenizer, config, analyzer)

spans  = curator.process_batch(texts)
final  = curator.prepare_final_dataset(spans)
curator.save_dataset(final, "outputs/curated/curated_spans.csv")
```

```python
# Topic-guided labeling
from emerge.src.curation.curator import Curator, CuratorConfig
from emerge.src.curation.analyzers.late_interaction_cross_encoder_analyzer import LateInteractionCrossEncoder, Topic
from emerge.src.curation.analyzers.base_analyzer import AnalyzerConfig

topics = [
    Topic(id=1, name="Customer Issues",
          description="Complaints/support requests",
          examples=["My product stopped working", "I can't access my account"]),
    Topic(id=2, name="Feature Requests",
          description="New features or improvements",
          examples=["Add dark mode", "Integrate with Slack"]),
]

analyzer = LateInteractionCrossEncoder(
    config=AnalyzerConfig(min_len=10, max_len=100),
    topics=topics,
    relevance_threshold=0.4,
    prediction_mode='late_chunking',
)

config  = CuratorConfig(min_length=15, max_length=150, quality_threshold=0.4, stride=2)
curator = Curator(model, tokenizer, config, analyzer)

labeled = curator.process_batch(customer_texts)
final   = curator.prepare_final_dataset(labeled)
curator.save_dataset(final, "outputs/curated/topic_spans.csv")
```

## 🔍 Example

Given a conversation about quarterly business performance:

```
Customer: How did we perform in Q3 overall?
Rep: Our Q3 results were strong. Revenue grew by 15% year-over-year, 
     and we exceeded our target profit margin by 2.3 percentage points. 
     The new product line contributed significantly, accounting for 
     about 30% of our growth this quarter.
     In other areas, we need to source more raw materials to keep up with demand.
     Supply chain issues from last year have mostly been resolved.
     Overall, it's been a solid quarter with positive momentum going into Q4.
```

Emerge identifies relevant semantic chunks and their clusters:

| Chunk | Cluster ID | Cluster Title | Relevance |
|-------|------------|---------------|-----------|
| "Revenue grew by 15% year-over-year" | 2451 | Financial Performance Metrics | 0.95 |
| "exceeded our target profit margin by 2.3 percentage points" | 2451 | Financial Performance Metrics | 0.87 |
| "new product line contributed significantly" | 3782 | Product Performance | 0.82 |
| "accounting for about 30% of our growth" | 2451 | Financial Performance Metrics | 0.79 |
| "Q3 results were strong" | 1123 | Quarterly Business Overview | 0.75 |
| "We need to source more raw materials" | 4890 | Supply Chain Management | 0.65 |
| "Supply chain issues from last year have mostly been resolved" | 4890 | Supply Chain Management | 0.60 |
| "Overall, it's been a solid quarter with positive momentum going into Q4" | 1123 | Quarterly Business Overview | 0.75 |

This structured output can then be used to train embedding models that better understand the semantic relationships between concepts like "revenue growth", "profit margins", and "product performance".


---

## Training Recipes

After exporting a dataset (columns like `text_span`, `cluster_id`/`topic_id`, `relevance`, …) choose one of the recipes below.

### Recipe A — Unsupervised clusters → Contrastive fine-tuning

Mine positives (same cluster) and negatives (other clusters). Sample by `relevance` to prioritize high-signal spans.

```python
import pandas as pd, random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load curated spans
df = pd.read_csv("outputs/curated/curated_spans.csv")
df = df.dropna(subset=["text_span", "cluster_id"]).copy()

# Keep top-K per cluster (optional)
K = 50
df.sort_values(["cluster_id", "relevance"], ascending=[True, False], inplace=True)
top = df.groupby("cluster_id").head(K)

# Build positive pairs
pairs = []
for _, g in top.groupby("cluster_id"):
    spans = g["text_span"].tolist()
    random.shuffle(spans)
    for i in range(0, len(spans)-1, 2):
        pairs.append(InputExample(texts=[spans[i], spans[i+1]]))

train_loader = DataLoader(pairs, batch_size=64, shuffle=True)
model = SentenceTransformer("thenlper/gte-base")        # choose a strong base
loss  = losses.MultipleNegativesRankingLoss(model)

model.fit(train_objectives=[(train_loader, loss)],
          epochs=2, warmup_steps=1000, show_progress_bar=True)

model.save("artifacts/models/gte-base-emerge-ft")
```

**Why it works** – spans in the same cluster are semantically close; MNR loss pulls positives together and pushes others apart.

### Recipe B — Topic-guided → Retrieval-style bi-encoder

Treat each topic as a “query” and its spans as “documents.”

```python
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

df = pd.read_csv("outputs/curated/topic_spans.csv")
# columns: text_span, topic_id, topic_name, topic_description, relevance

pairs = []
for _, row in df.iterrows():
    query = f"{row['topic_name']}: {row['topic_description']}"
    doc   = row["text_span"]
    pairs.append(InputExample(texts=[query, doc]))

train_loader = DataLoader(pairs, batch_size=32, shuffle=True)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
loss  = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(train_loader, loss)],
          epochs=2, warmup_steps=500, show_progress_bar=True)

model.save("artifacts/models/bge-topic-emerge-ft")
```

#### Optional: Triplet mining
- Positives: same cluster/topic  
- Negatives: different clusters/topics or low-relevance spans  
- Weight hard-negative sampling by `relevance`.

---

## Evaluation

1. **Retrieval sanity** – run topic queries, inspect top spans.  
2. **Task-level** – evaluate on your search/RAG datasets (hits@k, nDCG@k, MRR) comparing base vs fine-tuned.  
3. **Benchmark** – optional MTEB tasks relevant to your domain.

```python
from sentence_transformers import SentenceTransformer, util
import torch, pandas as pd

model = SentenceTransformer("artifacts/models/gte-base-emerge-ft")
corpus = pd.read_csv("outputs/curated/curated_spans.csv")["text_span"].tolist()
corpus_emb = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)

query = "Pricing inquiries: questions about plans and billing"
q_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

scores = util.cos_sim(q_emb, corpus_emb)[0]
best_idx = torch.topk(scores, k=10).indices.tolist()
for i in best_idx:
    print(corpus[i], float(scores[i]))
```

---

## When to use which analyzer

| Analyzer | Use-case | Outcome |
|----------|----------|---------|
| **LocalClusterAnalyzer** | No predefined labels; discover domain structure | Unsupervised clusters for contrastive fine-tuning |
| **LateInteractionCrossEncoder** | You have business topics/categories | Labeled spans for retrieval/classification training |

---

## Practical tips

- Use `relevance` to cap low-signal spans and balance clusters/topics.  
- Normalize embeddings (`normalize_embeddings=True`) during training/eval for stable cosine similarity.  
- Start with 1-2 epochs; compare base vs fine-tuned before scaling out.  
- Log dataset stats (clusters, spans/cluster, label balance) for reproducibility.  
