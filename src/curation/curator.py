from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from pathlib import Path
import logging
import json
from datetime import datetime
from tqdm import tqdm

from .utils.span_utils import find_chunk_spans
from .utils.late_chunking import late_chunking
from .utils.prediction_utils import get_embeddings, streaming_embeddings
from .analyzers.base_analyzer import BaseAnalyzer, AnalyzerConfig

logger = logging.getLogger(__name__)

@dataclass
class CuratorConfig:
    """Extended configuration for curation process"""
    # Basic curation settings
    min_length: int = 5
    max_length: int = 400
    batch_size: int = 32
    stride: int = 1
    top_k: int = 10
    
    # Column mappings (matching analyzer output)
    text_column: str = 'text'
    cluster_column: str = 'cluster'
    title_column: str = 'cluster_title'
    summary_column: str = 'cluster_summary'
    length_column: str = 'TURN_LEN'
    embedding_column: str = 'embedding'
    
    # Quality control
    quality_threshold: float = 0.5
    deduplication: bool = True
    min_cluster_size: int = 3
    
    # Processing settings
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Summarization settings
    summary_enabled: bool = True
    max_samples_per_cluster: int = 100
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class Curator:
    """Enhanced main class for data curation with batch processing and quality control"""
    
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        config: CuratorConfig,
        analyzer: BaseAnalyzer
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.analyzer = analyzer
        self.device = config.device
        
        # Initialize cache if enabled
        if config.cache_enabled:
            self.cache_dir = config.cache_dir or Path("cache/curator")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized cache at {self.cache_dir}")

    def process_batch(
        self, 
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Process texts in batches with progress tracking"""
        batch_size = batch_size or self.config.batch_size
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch = texts[i:i + batch_size]
            batch_results = self._process_single_batch(batch)
            results.extend(batch_results)
            
        return pd.DataFrame(results)

    def _process_single_batch(self, texts: List[str]) -> List[Dict]:
        """Process a single batch of texts"""
        batch_results = []
        
        for idx, text in enumerate(texts):
            try:
                sent_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}"
                prediction = self.curate_spans(text, sent_id)
                
                # Extract spans and metadata
                spans = self._extract_spans_from_prediction(prediction, text, sent_id)
                batch_results.extend(spans)
                
            except Exception as e:
                logger.error(f"Error processing text {idx}: {str(e)}")
                continue
                
        return batch_results

    def curate_spans(self, text: str, sent_id: str) -> Dict:
        """Curate spans from text using configured analyzer
        
        Args:
            text: Input text to process
            sent_id: Unique identifier for the text
            
        Returns:
            Dictionary containing:
                - representative_span: Most representative span
                - cluster_ids: List of cluster IDs
                - relevance_scores: List of relevance scores
                - cluster_metrics: DataFrame of cluster metrics
        """
        logger.debug(f"Curating spans for text {sent_id[:30]}...")
        
        try:
            # Use analyzer to predict spans
            representative_span, pareto_clusters, score_df, scores = self.analyzer.predict(
                sentence=text,
                stride=self.config.stride,
                min_len=self.config.min_length,
                max_len=self.config.max_length,
                sent_id=sent_id,
                cache_sent=self.config.cache_enabled,
                verbose=False,
                top_k=self.config.top_k
            )
            
            return {
                'representative_span': representative_span,
                'cluster_ids': pareto_clusters,
                'relevance_scores': scores,
                'cluster_metrics': score_df
            }
            
        except Exception as e:
            logger.error(f"Failed to curate spans for {sent_id}: {str(e)}")
            raise

    def _extract_spans_from_prediction(
        self, 
        prediction: Dict,
        original_text: str,
        sent_id: str
    ) -> List[Dict]:
        """Extract structured span information from prediction"""
        spans = []
        
        try:
            # Extract spans from analyzer results
            analyzer_results = self.analyzer.sentences[sent_id]['predicted_df']
            
            for cluster_id, score in zip(prediction['cluster_ids'], prediction['relevance_scores']):
                cluster_spans = analyzer_results[
                    analyzer_results[self.config.cluster_column] == cluster_id
                ]
                
                for _, span_row in cluster_spans.iterrows():
                    # Skip spans that don't meet quality threshold
                    if score < self.config.quality_threshold:
                        continue
                    
                    # Get text and ensure it's a string
                    span_text = str(span_row[self.config.text_column])
                    
                    # Create span info with correct column mappings
                    span_info = {
                        'text': span_text,
                        'cluster_id': int(cluster_id),
                        'relevance_score': float(score),
                        'text_length': int(span_row[self.config.length_column]),  # Use TURN_LEN
                        'original_text_id': sent_id,
                        'cluster_title': str(span_row[self.config.title_column]),
                        'cluster_summary': str(span_row[self.config.summary_column]),
                        'metadata': {
                            'original_text': original_text,
                            'span_position': original_text.find(span_text),
                            'embedding': span_row[self.config.embedding_column].tolist(),  # Convert numpy array to list
                            'cluster_metrics': prediction['cluster_metrics'].loc[cluster_id].to_dict() if cluster_id in prediction['cluster_metrics'].index else {},
                            'processed_at': datetime.now().isoformat()
                        }
                    }
                    spans.append(span_info)
                    
        except Exception as e:
            logger.error(f"Failed to extract spans from prediction: {str(e)}")
            logger.debug(f"Available columns: {analyzer_results.columns.tolist() if 'analyzer_results' in locals() else 'N/A'}")
            raise
                
        return spans

    def save_dataset(self, dataset: pd.DataFrame, path: Union[str, Path], include_metadata: bool = True) -> None:
        """Save curated dataset with metadata"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Make a copy for storage
        storage_df = dataset.copy()
        
        # Helper function to convert values to JSON-serializable format
        def serialize_value(v):
            if isinstance(v, Path):
                return str(v)
            elif isinstance(v, (np.int64, np.int32)):
                return int(v)
            elif isinstance(v, (np.float64, np.float32)):
                return float(v)
            elif isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, datetime):
                return v.isoformat()
            elif isinstance(v, (list, tuple)):
                return [serialize_value(x) for x in v]
            elif isinstance(v, dict):
                return {k: serialize_value(val) for k, val in v.items()}
            return v
        
        # Convert lists and numpy arrays to JSON-serializable format
        if 'CLUSTER_IDS' in dataset.columns:
            list_columns = ['CLUSTER_IDS', 'RELEVANCE_SCORES', 'CLUSTER_TITLES', 'CLUSTER_DESCRIPTIONS']
            for col in list_columns:
                # Ensure column values are serialized properly
                storage_df[col] = storage_df[col].apply(
                    lambda x: json.dumps(serialize_value(x))
                )
        
        # Save to parquet
        storage_df.to_parquet(save_path.with_suffix('.parquet'))
        
        # Convert config to serializable format
        config_dict = {
            k: serialize_value(v) 
            for k, v in self.config.__dict__.items()
        }
        
        # Prepare metadata with serializable values based on dataset format
        metadata = {
            'created_at': datetime.now().isoformat(),
            'config': config_dict,
            'stats': {}
        }
        
        # Add appropriate statistics based on dataset format
        if 'CLUSTER_IDS' in dataset.columns:
            # Final format dataset
            metadata['stats'].update({
                'total_texts': int(len(dataset)),
                'unique_clusters': len(set(serialize_value(cluster_id) for clusters in dataset['CLUSTER_IDS'] for cluster_id in clusters)),
                'avg_clusters_per_text': float(dataset['CLUSTER_IDS'].apply(len).mean()),
                'avg_relevance_score': float(np.mean([serialize_value(score) for scores in dataset['RELEVANCE_SCORES'] for score in scores]))
            })
        elif 'cluster_id' in dataset.columns:
            # Detailed format dataset
            metadata['stats'].update({
                'total_spans': int(len(dataset)),
                'unique_clusters': int(dataset['cluster_id'].nunique()),
                'avg_score': float(dataset['relevance_score'].mean())
            })
        
        metadata['save_path'] = str(save_path)
        
        # Save metadata if requested
        if include_metadata:
            with open(save_path.with_suffix('.json'), 'w') as f:
                json.dump(
                    metadata, 
                    f, 
                    indent=2,
                    default=serialize_value
                )
                
        logger.info(f"Saved dataset to {save_path}")

    @classmethod
    def load_dataset(
        cls,
        path: Union[str, Path],
        validate: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """Load curated dataset with validation"""
        load_path = Path(path)
        
        # Load dataset
        dataset = pd.read_parquet(load_path.with_suffix('.parquet'))
        
        # Convert string-encoded lists back to actual lists
        if 'CLUSTER_IDS' in dataset.columns:
            list_columns = ['CLUSTER_IDS', 'RELEVANCE_SCORES', 'CLUSTER_TITLES', 'CLUSTER_DESCRIPTIONS']
            for col in list_columns:
                dataset[col] = dataset[col].apply(json.loads)
        
        # Load metadata if exists
        metadata = {}
        metadata_path = load_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                
        # Validate if requested
        if validate:
            cls._validate_dataset(dataset)
            
        return dataset, metadata

    @staticmethod
    def _validate_dataset(dataset: pd.DataFrame) -> None:
        """Validate dataset structure and content for both detailed and final formats
        
        Args:
            dataset: DataFrame to validate
            
        Raises:
            ValueError: If dataset is invalid or missing required columns
        """
        # Check if empty
        if dataset.empty:
            raise ValueError("Dataset is empty")

        # Determine format based on columns
        if 'CLUSTER_IDS' in dataset.columns:
            # Final format validation
            required_columns = [
                'TEXT', 
                'CLUSTER_IDS', 
                'RELEVANCE_SCORES', 
                'CLUSTER_TITLES', 
                'CLUSTER_DESCRIPTIONS'
            ]
            
            for col in required_columns:
                if col not in dataset.columns:
                    raise ValueError(f"Missing required column in final format: {col}")
                    
            # Validate data types and content
            if dataset['TEXT'].isnull().any():
                raise ValueError("Dataset contains null text values")
                
            if not all(isinstance(x, list) for x in dataset['CLUSTER_IDS']):
                raise ValueError("CLUSTER_IDS must be a list for each row")
                
            if not all(isinstance(x, list) for x in dataset['RELEVANCE_SCORES']):
                raise ValueError("RELEVANCE_SCORES must be a list for each row")
                
        elif 'cluster_id' in dataset.columns:
            # Detailed format validation
            required_columns = ['text', 'cluster_id', 'relevance_score']
            
            for col in required_columns:
                if col not in dataset.columns:
                    raise ValueError(f"Missing required column in detailed format: {col}")
                    
            # Validate data types and content
            if dataset['text'].isnull().any():
                raise ValueError("Dataset contains null text values")
                
            if not pd.api.types.is_numeric_dtype(dataset['cluster_id']):
                raise ValueError("cluster_id must be numeric")
                
            if not pd.api.types.is_numeric_dtype(dataset['relevance_score']):
                raise ValueError("relevance_score must be numeric")
        else:
            raise ValueError("Unknown dataset format - missing required columns for either format")

    def compute_quality_metrics(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Compute quality metrics using analyzer scores and metadata
        
        Args:
            dataset: DataFrame with spans and their metadata
            
        Returns:
            DataFrame with added quality metrics
        """
        metrics = dataset.copy()
        
        # Basic span-level metrics
        metrics['relevance_score'] = metrics['relevance_score'].fillna(0.0).astype(float)
        metrics['text_coverage'] = metrics['text_length'] / metrics['metadata'].apply(
            lambda x: len(x['original_text'].split())
        )
        
        # Cluster-level aggregations
        cluster_stats = metrics.groupby('cluster_id').agg({
            'text': 'count',  # Spans per cluster
            'relevance_score': 'mean',  # Average quality
            'text_coverage': 'mean',  # Average coverage
        }).rename(columns={
            'text': 'cluster_size',
            'relevance_score': 'cluster_avg_score',
            'text_coverage': 'cluster_avg_coverage'
        })
        
        return metrics.merge(cluster_stats, on='cluster_id')

    def prepare_final_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Transform processed dataset into final format with required columns"""
        
        # Create a dictionary to store the final data
        final_data = {
            'TEXT': [],
            'CLUSTER_IDS': [], 
            'RELEVANCE_SCORES': [],
            'CLUSTER_TITLES': [],
            'CLUSTER_DESCRIPTIONS': []
        }
        
        # Process each unique original_text_id
        for text_id, group in dataset.groupby('original_text_id'):
            # Get original text from first row
            original_text = group.iloc[0]['metadata']['original_text']
            
            # Collect unique clusters while maintaining order of appearance in the analyzer results
            clusters_with_metadata = []
            seen_clusters = set()
            
            # Use the original order of prediction['cluster_ids'] from curate_spans if possible
            # Otherwise fall back to ordered appearance in the dataset
            for _, row in group.iterrows():
                cluster_id = row['cluster_id']
                if cluster_id not in seen_clusters:
                    seen_clusters.add(cluster_id)
                    clusters_with_metadata.append({
                        'cluster_id': int(cluster_id),
                        'relevance_score': float(row['relevance_score']),
                        'cluster_title': str(row['cluster_title']),
                        'cluster_summary': str(row['cluster_summary'])
                    })
            
            # Add to final dataset
            final_data['TEXT'].append(original_text)
            final_data['CLUSTER_IDS'].append([item['cluster_id'] for item in clusters_with_metadata])
            final_data['RELEVANCE_SCORES'].append([item['relevance_score'] for item in clusters_with_metadata])
            final_data['CLUSTER_TITLES'].append([item['cluster_title'] for item in clusters_with_metadata])
            final_data['CLUSTER_DESCRIPTIONS'].append([item['cluster_summary'] for item in clusters_with_metadata])
        
        # Create DataFrame
        final_dataset = pd.DataFrame(final_data)
        
        # Verify lengths match
        for _, row in final_dataset.iterrows():
            cluster_len = len(row['CLUSTER_IDS'])
            assert len(row['RELEVANCE_SCORES']) == cluster_len, "Mismatch in CLUSTER_IDS and RELEVANCE_SCORES lengths"
            assert len(row['CLUSTER_TITLES']) == cluster_len, "Mismatch in CLUSTER_IDS and CLUSTER_TITLES lengths"
            assert len(row['CLUSTER_DESCRIPTIONS']) == cluster_len, "Mismatch in CLUSTER_IDS and CLUSTER_DESCRIPTIONS lengths"
        
        return final_dataset


if __name__ == '__main__':
    import os
    import torch
    import pandas as pd
    from pathlib import Path
    from datetime import datetime
    from transformers import AutoTokenizer, AutoModel
    from src.embedding.clustering import ClusteringConfig, ClusteringPipeline
    from src.curation.analyzers.local_cluster_analyzer import LocalClusterAnalyzer
    from src.curation.analyzers.base_analyzer import AnalyzerConfig
    from src.curation.curator import Curator, CuratorConfig
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting curator demo")
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
    model = AutoModel.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True, 
        device_map='cpu', 
        torch_dtype=torch.float16
    )
    logger.info(f"Base model initialized on device: {model.device}")

    # 2. Load Clustering Pipeline
    logger.info("Loading clustering pipeline...")
    try:
        pipeline, summaries = ClusteringPipeline.load("artifacts/clustering/models")
        logger.info(f"Pipeline loaded successfully (fitted: {pipeline._is_fitted})")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {str(e)}")
        raise

    # 3. Initialize Analyzer
    logger.info("Setting up local cluster analyzer...")
    analyzer = LocalClusterAnalyzer(
        pipeline=pipeline,
        config=AnalyzerConfig(
            min_len=5,
            max_len=50,
            top_k=30,
            stride=1,
            clustering_config=pipeline.config
        ),
        cache_save_dir='cache/local_analyzer',
        model=model,
        tokenizer=tokenizer,
        model_kwargs={
            'model_batch_size': 32,
            'lora_task': 'separation'
        }
    )

    # 4. Configure Curator
    logger.info("Initializing curator...")
    config = CuratorConfig(
        # Basic settings
        min_length=10,
        max_length=200,
        batch_size=32,
        quality_threshold=0.6,
        
        # Column mappings (matching analyzer output)
        text_column='text',
        cluster_column='cluster',
        title_column='cluster_title',
        summary_column='cluster_summary',
        length_column='TURN_LEN',
        embedding_column='embedding',
        
        # Cache settings
        cache_enabled=True,
        cache_dir=Path("cache/curator"),
        
        # Summary settings
        summary_enabled=True,
        max_samples_per_cluster=100,
        
        # Metadata
        metadata={
            'model_name': MODEL_NAME,
            'pipeline_version': '1.0',
            'created_at': datetime.now().isoformat()
        }
    )

    # 5. Initialize Curator
    try:
        curator = Curator(
            model=model,
            tokenizer=tokenizer,
            config=config,
            analyzer=analyzer
        )
        logger.info("Curator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize curator: {str(e)}")
        raise

    # 6. Load Test Data
    logger.info("Loading test corpus...")
    try:
        test_corpus = pd.read_csv("src/curation/analyzers/sample_texts.csv")
        texts = test_corpus['text'].tolist()[:2]
        labels = test_corpus['topic'].tolist()[:2]
        
        logger.info(f"Loaded {len(texts)} texts for processing")
        logger.debug(f"Sample text length: {len(texts[0].split())} words")
        logger.debug(f"Sample label: {labels[0]}")
    except Exception as e:
        logger.error(f"Failed to load test corpus: {str(e)}")
        raise

    # 7. Process Texts
    logger.info("Processing texts in batches...")
    try:
        dataset = curator.process_batch(
            texts=texts,
            batch_size=config.batch_size
        )
        
        if len(dataset) == 0:
            logger.warning("No valid spans were extracted from the texts")
            logger.info("=" * 80)
            logger.info("Curator demo completed with no results")
            exit(0)
            
        logger.info(f"Successfully processed {len(dataset)} spans")
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise

    # 8. Compute Quality Metrics
    logger.info("Computing quality metrics...")
    try:
        dataset_with_metrics = curator.compute_quality_metrics(dataset)
        metrics_summary = dataset_with_metrics.agg({
            'text_coverage': ['mean', 'std'],
            'relevance_score': ['mean', 'std']
        })
        logger.info("\nQuality Metrics Summary:")
        logger.info(f"\n{metrics_summary}")
        
        # Transform to final format
        logger.info("Preparing final dataset...")
        final_dataset = curator.prepare_final_dataset(dataset_with_metrics)
        
    except Exception as e:
        logger.error(f"Failed to compute metrics or prepare final dataset: {str(e)}")
        raise

    # 9. Save Results
    output_path = Path("outputs/curated_datasets")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_path / f"curated_dataset_{timestamp}"

    logger.info(f"Saving results to {save_path}...")
    try:
        # Save both datasets - detailed for debugging and final for use
        curator.save_dataset(
            dataset=final_dataset,  # Save the simplified version as main dataset
            path=save_path,
            include_metadata=True
        )
        # Optionally save detailed version with _full suffix
        curator.save_dataset(
            dataset=dataset_with_metrics,
            path=save_path.with_name(f"{save_path.name}_full"),
            include_metadata=True
        )
        logger.info("Datasets saved successfully")
    except Exception as e:
        logger.error(f"Failed to save dataset: {str(e)}")
        raise

    # 10. Validation
    logger.info("Validating saved dataset...")
    try:
        loaded_dataset, metadata = Curator.load_dataset(
            path=save_path,
            validate=True
        )
        logger.info("Dataset validation successful")
        logger.info("\nDataset Statistics:")
        logger.info(f"Total spans: {len(loaded_dataset)}")
        logger.info(f"Unique clusters: {loaded_dataset['cluster_id'].nunique()}")
        logger.info(f"Average relevance score: {loaded_dataset['relevance_score'].mean():.3f}")
    except Exception as e:
        logger.error(f"Dataset validation failed: {str(e)}")
        raise

    logger.info("=" * 80)
    logger.info("Curator demo completed successfully")