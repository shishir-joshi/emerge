import torch
import pandas as pd
from src.curation.analyzers.base_analyzer import BaseAnalyzer, AnalyzerConfig
from src.curation.analyzers.cluster_analyzer import ClusterAnalyzer
# from src.curation.curator import Curator
from src.embedding.clustering import ClusteringConfig, ClusteringPipeline


class LocalClusterAnalyzer(ClusterAnalyzer):
    """Analyzer for local embedding model based span prediction."""
    def __init__(self, pipeline: ClusteringPipeline, config: AnalyzerConfig, cache_save_dir=None, model=None, tokenizer=None, model_kwargs={}):
        super().__init__(pipeline, config, cache_save_dir)
        # self.pipeline = pipeline
        # self.pipeline_dims = len(pipeline.config.embedding_dims)
        
        self.tokenizer = tokenizer
        self.model = model
        self._setup_model(model_kwargs)
        
    def _setup_model(self, model_kwargs):
        self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.model_batch_size = model_kwargs['model_batch_size']
        self.task = model_kwargs['lora_task'] # 'separation' or 'retrieval.passage'

        # Prepare adapter_mask and move to GPU
        if self.task:
            self.task_id = self.model._adaptation_map[self.task]
            self.adapter_mask = torch.full(
                (1,), self.task_id, dtype=torch.int32
            ).to(self.device)
            
    def embed_chunks(self, chunks):
        # chunk_embeddings = get_embeddings(filtered_chunks, self.model, self.tokenizer, device=self.device, task=self.task, batch_size=self.model_batch_size)
        chunk_embeddings = self.model.encode(chunks, task=self.task, batch_size=self.model_batch_size)
        return chunk_embeddings


    def predict(self, sentence, stride, min_len, max_len, sent_id, cache_sent=False, verbose=False, top_k=10):
        self.logger.info("---------------------------------------------------")
        if sent_id not in self.sentences:
            self.logger.info(f"Predicting clusters for sentence ID {sent_id}.")
            self.config.min_len = min_len
            self.config.max_len = max_len

            self.logger.info(f"Creating overlapping chunks with stride {stride}.")
            chunks = self.create_overlapping_chunks_with_stride(sentence, stride)

            self.logger.info(f"Filtering chunks by length.")
            filtered_chunks = [chunk for chunk in chunks if min_len <= len(chunk.split()) <= max_len]
            
            # Direct Embedding
            self.logger.info(f"Embed All chunks directly:")
            chunk_embeddings = self.embed_chunks(filtered_chunks)

            self.logger.info(f"Predicting clusters for filtered chunks.")
            cluster_predictions = self.predict_clusters(filtered_chunks, chunk_embeddings, verbose=verbose)

            if cache_sent:
                self.sentences[sent_id] = {
                    'sentence': sentence,
                    'chunks': filtered_chunks,
                    'chunks_embs': chunk_embeddings,
                    'predicted_df': cluster_predictions
                }
        else:
            filtered_chunks = self.sentences[sent_id]['chunks']
            chunk_embeddings = self.sentences[sent_id]['chunks_embs']
            cluster_predictions = self.sentences[sent_id]['predicted_df']

        self.logger.info(f"Filtering out noise clusters.")
        cluster_predictions = cluster_predictions[cluster_predictions['cluster'] != -1]
        
        # If all noise, skip
        if cluster_predictions.empty:
            self.logger.info(f"No clusters detected, womp womp")
            return pd.DataFrame(), [], pd.DataFrame(), []


        self.logger.info(f"Identifying top {top_k} Pareto optimal clusters.")
        if min_len is not None:
            cluster_predictions = cluster_predictions[cluster_predictions['TURN_LEN'] >= min_len]
        if max_len is not None:
            cluster_predictions = cluster_predictions[cluster_predictions['TURN_LEN'] <= max_len]
            
        pareto_cluster_ids, score_df, scores = self._get_pareto_optimal_clusters(
            cluster_predictions,
            sentence,
            top_k=top_k
        )

        self.logger.info(f"Getting the most representative span (longest span) from the top cluster.")
        representative_span = cluster_predictions[cluster_predictions['cluster'] == pareto_cluster_ids[0]]
        representative_span = representative_span.loc[representative_span['TURN_LEN'].idxmax()]
        self.logger.info("---------------------------------------------------")
        return representative_span.to_frame().T, pareto_cluster_ids, score_df, scores
    

if __name__ == "__main__":
    import os
    import torch
    import pandas as pd
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    from src.embedding.clustering import ClusteringConfig, ClusteringPipeline
    from src.curation.curator import AnalyzerConfig
    from src.curation.analyzers.local_cluster_analyzer import LocalClusterAnalyzer
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting local cluster analyzer demo")
    logger.info("=" * 80)

    # 1. Model Loading
    MODEL_NAME = "jinaai/jina-embeddings-v3"
    logger.info(f"Initializing with model: {MODEL_NAME}")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    logger.debug("Tokenizer configuration loaded successfully")

    logger.info("Loading base model...")
    base_model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, device_map='cpu', torch_dtype=torch.float16
    )
    logger.info(f"Base model initialized on device: {base_model.device}")

    # 2. Pipeline Setup
    logger.info("Initializing clustering pipeline...")
    try:
        pipeline, summaries = ClusteringPipeline.load("artifacts/clustering/models")
        logger.info(f"Pipeline loaded successfully (fitted: {pipeline._is_fitted})")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {str(e)}")
        raise

    # 3. Configuration
    logger.info("Setting up curation configuration...")
    curation_config = AnalyzerConfig(
        min_len=5,
        max_len=50,
        top_k=30,
        stride=1,
        clustering_config=pipeline.config
    )
    logger.debug(f"Curation config: min_len={curation_config.min_len}, "
                f"max_len={curation_config.max_len}, top_k={curation_config.top_k}")

    # 4. Model Parameters
    logger.info("Configuring model parameters...")
    model_kwargs = {
        'model_batch_size': 32,
        'lora_task': 'separation' #'retrieval.passage'
    }
    logger.debug(f"Model parameters: {model_kwargs}")

    # 5. Analyzer Initialization
    logger.info("Initializing local analyzer...")
    try:
        analyzer = LocalClusterAnalyzer(
            pipeline=pipeline,
            config=curation_config,
            cache_save_dir='cache/local_analyzer',
            model=base_model,
            tokenizer=tokenizer,
            model_kwargs=model_kwargs
        )
        logger.info("Analyzer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {str(e)}")
        raise

    # 6. Test Data Preparation
    logger.info("Loading test corpus...")
    try:
        test_corpus = pd.read_csv("src/curation/analyzers/sample_texts.csv")
        text = test_corpus['text'].values[3]
        label = test_corpus['topic'].values[3]
        
        logger.debug(f"Selected text length: {len(text.split())} words")
        logger.debug(f"Text label: {label}")
    except Exception as e:
        logger.error(f"Failed to load test corpus: {str(e)}")
        raise

    # 7. Prediction
    logger.info("Running prediction pipeline...")
    try:
        representative_span, pareto_clusters, score_df, scores = analyzer.predict(
            sentence=text,
            stride=1,
            min_len=10,
            max_len=30,
            sent_id="example_1",
            cache_sent=True,
            verbose=True,
            top_k=3
        )
        logger.info("Prediction completed successfully")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

    # 8. Results Analysis
    logger.info("Analyzing results...")
    
    # Representative span analysis
    rep_span_text = representative_span[analyzer.config.clustering_config.text_column].values[0]
    logger.info(f"Representative span identified: {rep_span_text}")

    # Cluster analysis
    logger.info(f"Found {len(pareto_clusters)} Pareto optimal clusters")
    for i, cluster_id in enumerate(pareto_clusters):
        cluster_spans = analyzer.sentences["example_1"]['predicted_df'][
            analyzer.sentences["example_1"]['predicted_df']['cluster'] == cluster_id
        ]
        
        logger.info(f"\nCluster {i+1} (ID: {cluster_id}):")
        logger.info(f"Title: {summaries[str(cluster_id)]['cluster_title']}")
        logger.info(f"Summary: {summaries[str(cluster_id)]['summary']}")
        logger.info(f"Spans in cluster: {len(cluster_spans)}")

    logger.info("=" * 80)
    logger.info("Analysis completed successfully")
