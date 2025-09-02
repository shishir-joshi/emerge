# Emerge

## What is this?

A text chunking and clustering tool. It takes your text, breaks it into overlapping chunks, embeds them, clusters similar chunks together, and gives you back a labeled dataset.

That's it. No magic, no AI buzzwords.

## What it actually does

1. **Chunks your text** - Splits text into overlapping spans of different lengths
2. **Embeds the chunks** - Uses Jina embeddings to convert text chunks to vectors  
3. **Clusters similar chunks** - Groups chunks using UMAP + HDBSCAN clustering
4. **Labels everything** - Assigns cluster IDs and generates cluster titles via OpenAI API
5. **Saves a dataset** - Outputs CSV with chunks and their cluster assignments

Useful if you want to automatically organize and label chunks of text for dataset creation.

## How it works

Basic pipeline:

```
Text → Chunk → Embed → Cluster → Label → Save
```

1. **Chunk**: Create overlapping text spans (configurable min/max length)
2. **Embed**: Convert chunks to vectors using Jina embeddings  
3. **Cluster**: Group similar chunks with UMAP+HDBSCAN
4. **Label**: Generate cluster titles using OpenAI API
5. **Save**: Output CSV with chunks and cluster assignments

## Example output

Input text:
```
"Our Q3 results were strong. Revenue grew by 15% year-over-year, 
and we exceeded our target profit margin by 2.3 percentage points."
```

After processing, you get chunks labeled by cluster:
```
Chunk: "Revenue grew by 15% year-over-year" → Cluster 2451 (Financial Metrics)
Chunk: "exceeded our target profit margin" → Cluster 2451 (Financial Metrics)  
Chunk: "Q3 results were strong" → Cluster 1892 (Performance Summary)
```

## What's in the code

**Curator** (`src/curation/curator.py`)
- Main class that orchestrates the whole pipeline
- Processes batches of text and saves datasets

**LocalClusterAnalyzer** (`src/curation/analyzers/local_cluster_analyzer.py`)  
- Handles the chunking and cluster assignment
- Uses pre-trained clustering pipeline

**ClusteringPipeline** (`src/embedding/clustering.py`)
- Does the UMAP + HDBSCAN clustering
- Needs to be trained/loaded from artifacts

**ClusterSummarizer** (`src/embedding/cluster_summarizer.py`)
- Generates cluster titles using OpenAI API

## Setup

You need a pre-trained clustering pipeline in `artifacts/clustering/models/` for this to work. 

```bash
git clone https://github.com/shishir-joshi/emerge.git
cd emerge
pip install -r requirements.txt
```

## Basic usage

```python
from src.curation.curator import Curator, CuratorConfig
from src.curation.analyzers.local_cluster_analyzer import LocalClusterAnalyzer
from src.curation.analyzers.base_analyzer import AnalyzerConfig
from src.embedding.clustering import ClusteringPipeline
from transformers import AutoTokenizer, AutoModel

# Load your model
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

# Load clustering pipeline (you need this pre-trained)
pipeline, _ = ClusteringPipeline.load("artifacts/clustering/models")

# Set up analyzer
analyzer = LocalClusterAnalyzer(
    pipeline=pipeline,
    config=AnalyzerConfig(min_len=5, max_len=50),
    model=model,
    tokenizer=tokenizer,
    model_kwargs={'model_batch_size': 32, 'lora_task': 'separation'}
)

# Set up curator
config = CuratorConfig(min_length=10, max_length=200)
curator = Curator(model, tokenizer, config, analyzer)

# Process your texts
texts = ["Your text here", "Another text"]
dataset = curator.process_batch(texts)
final_dataset = curator.prepare_final_dataset(dataset)

# Save results
curator.save_dataset(final_dataset, "output/my_dataset")
```

Run the demo:
```bash
python src/curation/curator.py
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers (for Jina embeddings)
- UMAP, HDBSCAN (for clustering)
- OpenAI API key (for cluster naming)
- Pre-trained clustering pipeline (not included)

## License

MIT License - see LICENSE file.
