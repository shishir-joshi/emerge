from IPython.display import HTML, Javascript, display
from datetime import datetime
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
from itables import show
from src.curation.utils.async_embedding_client import embed
from src.curation.analyzers.base_analyzer import BaseAnalyzer, AnalyzerConfig
from src.embedding.clustering import ClusteringPipeline, ClusteringConfig


class ClusterAnalyzer(BaseAnalyzer):
    def __init__(self, pipeline: ClusteringPipeline, config: AnalyzerConfig, cache_save_dir=None):
        super().__init__(config, cache_save_dir)
        self.pipeline = pipeline
        self.pipeline_dims = pipeline.config.embedding_dims

    def embed_chunks(self, chunks):
        chunks_embs = embed(
            chunks, 
            embedding_model='text-embedding-3-small', 
            dimensions=self.pipeline_dims
        )
        return [i[1][0] for i in chunks_embs]

    def predict_clusters(self, chunks, chunks_embs, verbose=False):
        chunks_df = pd.DataFrame(
            [[chunks, chunks_embs]], 
            columns=[self.pipeline.config.text_column, self.pipeline.config.embedding_column]
        )

        chunks_df = chunks_df.explode([self.pipeline.config.text_column, self.pipeline.config.embedding_column])
        chunks_df.reset_index(drop=True, inplace=True)

        chunks_df['TURN_LEN'] = chunks_df[self.pipeline.config.text_column].str.split(' ').str.len()

        chunks_df_min_len = chunks_df[
            (chunks_df[self.pipeline.config.text_column].str.split(" ").str.len() >= self.config.min_len)
            & (chunks_df[self.pipeline.config.text_column].str.split(" ").str.len() <= self.config.max_len)
        ]
        
        return self.pipeline.predict_new_data(chunks_df_min_len)


if __name__ == "__main__":
    # Usage
    from src.embedding.pipeline import EmbeddingPipeline
    from src.embedding.clustering import ClusteringConfig

    # Create sample configuration
    config = ClusteringConfig(
        min_cluster_size=5,
        min_samples=5,
        umap_n_components=50
    )

    # Initialize pipeline
    pipeline = EmbeddingPipeline(
        model_name='all-MiniLM-L6-v2',
        config=config
    )

    # Initialize analyzer
    config = AnalyzerConfig(min_len=5, max_len=400, base_color=(255, 105, 180))
    analyzer = ClusterAnalyzer(pipeline, config)

    # Analyze a sentence
    sentence = "The quick brown fox jumps over the lazy dog."
    stride = 1
    min_len = 40
    max_len = 100
    sent_id = 0
    cache_sent = False
    verbose = True
    top_k = 10
    representative_span, pareto_cluster_ids, score_df, scores = analyzer.predict(
        sentence, stride, min_len, max_len, sent_id, cache_sent, verbose, top_k
    )
    print(representative_span)
    print(pareto_cluster_ids)
    
