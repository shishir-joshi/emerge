from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional
import numpy as np
import pandas as pd
from umap import UMAP
from umap.parametric_umap import ParametricUMAP
import hdbscan
from itables import show
from sklearn.metrics import silhouette_score
import logging
import pickle
import os
from pathlib import Path
from datetime import datetime
import json

MODEL_EMBEDDING_DIMS = 768

@dataclass
class ClusteringConfig:
    """Configuration for clustering parameters"""
    # Input embedding dimensions
    embedding_dims: int = MODEL_EMBEDDING_DIMS
    text_column: str = 'text'
    embedding_column: str = 'embedding'
    cluster_labels_column: str = 'cluster'
    cluster_title_column: str = 'cluster_title'
    cluster_summary_column: str = 'cluster_summary'

    # UMAP parameters
    umap_n_neighbors: List[int] = field(default_factory=lambda: [5, 10, 15])
    umap_n_components: List[int] = field(default_factory=lambda: [50, 100])
    umap_min_dist: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.5])
    umap_metric: str = 'cosine'
    umap_parametric: bool = False
    
    # HDBSCAN parameters
    min_cluster_sizes: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    min_samples: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    cluster_selection_epsilon: List[float] = field(default_factory=lambda: [0.0, 0.1])
    cluster_selection_method: List[str] = field(default_factory=lambda: ['eom', 'leaf'])
    
    # Evaluation weights
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        'coherence_score': 2.0,
        'silhouette_score': 1.0,
        'relative_validity_score': 1.0,
        'noise_ratio': -1.0
    })

class ClusteringPipeline:
    """Main clustering pipeline implementing the topic clustering approach"""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.umap_model = None
        self.hdbscan_model = None
        self.best_params = None
        self.evaluation_results = None
        self._is_fitted = False
        
    @property
    def is_fitted(self) -> bool:
        """Check if pipeline is fitted"""
        return self._is_fitted and self.umap_model is not None and self.hdbscan_model is not None
    
    def save(self, save_path: str, summaries: Dict = None) -> None:
        """Save trained pipeline and summaries to disk
        
        Args:
            save_path: Directory path to save pipeline artifacts
            summaries: Optional dictionary containing cluster summaries
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
            
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle UMAP model saving separately
        umap_path = save_dir / 'umap_model'
        if self.config.umap_parametric:
            self.umap_model.save(str(umap_path))
            umap_save_info = {
                'type': 'parametric',
                'path': 'umap_model'
            }
        else:
            with open(umap_path, 'wb') as f:
                pickle.dump(self.umap_model, f)
            umap_save_info = {
                'type': 'standard',
                'path': 'umap_model'
            }
        
        # Save summaries if provided
        if summaries:
            summaries_path = save_dir / 'cluster_summaries.json'
            with open(summaries_path, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)
        
        # Save pipeline state
        state = {
            'config': self.config,
            'best_params': self.best_params,
            'evaluation_results': self.evaluation_results,
            'umap_info': umap_save_info,  # Save UMAP metadata
            'hdbscan_model': self.hdbscan_model,
            '_is_fitted': self._is_fitted,
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'version': '1.0',
                'has_summaries': summaries is not None
            }
        }
        
        # Save main state to disk
        with open(save_dir / 'pipeline_state.pkl', 'wb') as f:
            pickle.dump(state, f)
            
        logging.info(f"Saved pipeline state to {save_dir}")

    @classmethod
    def load(cls, load_path: str) -> Tuple['ClusteringPipeline', Optional[Dict]]:
        """Load trained pipeline and summaries from disk
        
        Args:
            load_path: Directory path containing saved pipeline artifacts
            
        Returns:
            Tuple of (ClusteringPipeline instance, Optional[Dict] of summaries)
        """
        load_dir = Path(load_path)
        if not load_dir.exists():
            raise FileNotFoundError(f"Save directory {load_dir} not found")
            
        # Load main state
        with open(load_dir / 'pipeline_state.pkl', 'rb') as f:
            state = pickle.load(f)
            
        # Create new pipeline instance
        pipeline = cls(state['config'])
        
        # Load UMAP model based on type
        umap_info = state['umap_info']
        umap_path = load_dir / umap_info['path']
        
        if umap_info['type'] == 'parametric':
            from umap.parametric_umap import load_ParametricUMAP
            pipeline.umap_model = load_ParametricUMAP(str(umap_path))
        else:
            with open(umap_path, 'rb') as f:
                pipeline.umap_model = pickle.load(f)
        
        # Restore other state
        pipeline.hdbscan_model = state['hdbscan_model']
        pipeline.best_params = state['best_params']
        pipeline.evaluation_results = state['evaluation_results']
        pipeline._is_fitted = state['_is_fitted']
        
        # Load summaries if they exist
        summaries = None
        if state['metadata'].get('has_summaries', False):
            summaries_path = load_dir / 'cluster_summaries.json'
            if summaries_path.exists():
                with open(summaries_path, 'r', encoding='utf-8') as f:
                    summaries = json.load(f)
        
        # Set summaries attribute if they exist
        if summaries is not None:
            pipeline.cluster_summaries = summaries
        else:
            pipeline.cluster_summaries = {}
            
        logging.info(f"Loaded pipeline state from {load_dir}")
        return pipeline, summaries
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Use fitted pipeline for inference on new data
        
        Args:
            embeddings: Input embeddings array of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Cluster labels for each input sample
        
        Raises:
            ValueError: If pipeline is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
            
        # Apply UMAP reduction
        reduced_embeddings = self.umap_model.transform(embeddings)
        
        # Apply HDBSCAN clustering with prediction data
        self.hdbscan_model.prediction_data = True
        cluster_labels = hdbscan.approximate_predict(
            self.hdbscan_model, 
            reduced_embeddings
        )[0]  # Only take labels, ignore probabilities
        
        return cluster_labels

    def predict_new_data(self, new_data: pd.DataFrame, text_column: str = None, embedding_column: str = None) -> pd.DataFrame:
        """Predict clusters for new data and add cluster information
        
        Args:
            new_data: DataFrame containing new data
            text_column: Optional name of text column (if different from config)
            embedding_column: Optional name of embedding column (if different from config)
            
        Returns:
            pd.DataFrame: Input DataFrame with added cluster predictions and metadata
        """
        # Use configured column names if not specified
        text_column = text_column or self.config.text_column
        embedding_column = embedding_column or self.config.embedding_column
        
        # Extract embeddings
        embeddings = np.stack(new_data[embedding_column].values)
        
        # Get cluster assignments
        labels = self.predict(embeddings)
        
        # Add predictions to dataframe
        results_df = new_data.copy()
        results_df['cluster'] = labels
        
        # Add cluster metadata if summaries exist
        if hasattr(self, 'cluster_summaries') and self.cluster_summaries:
            results_df['cluster_title'] = results_df['cluster'].map(
                lambda x: self.cluster_summaries[str(x)]['cluster_title'] 
                if str(x) in self.cluster_summaries and x != -1 
                else "Noise"
            )
            results_df['cluster_summary'] = results_df['cluster'].map(
                lambda x: self.cluster_summaries[str(x)]['summary']
                if str(x) in self.cluster_summaries and x != -1
                else "Noise points"
            )
            
        return results_df
    
    def fit_transform(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Run the complete clustering pipeline"""
        # Run hyperparameter search
        best_params, evaluation_results = self._hyperparameter_search(embeddings)
        self.best_params = best_params
        self.evaluation_results = evaluation_results
        
        # Train final models with best parameters
        print(f"Training final models with best parameters: {best_params}")
        reduced_embeddings = self._fit_umap(embeddings, best_params['umap_params'])
        cluster_labels = self._fit_hdbscan(reduced_embeddings, best_params['hdbscan_params'])
        
        self._is_fitted = True
        return cluster_labels, evaluation_results

    def _hyperparameter_search(self, embeddings: np.ndarray) -> Tuple[Dict, pd.DataFrame]:
        """Run hyperparameter search and return best parameters"""
        results = []
        
        # Generate parameter combinations
        umap_params = self._generate_umap_params()
        hdbscan_params = self._generate_hdbscan_params()
        
        for u_params in umap_params:
            print(f"Running UMAP with params: {u_params}")
            reduced_data = self._fit_umap(embeddings, u_params)
            
            for h_params in hdbscan_params:
                try:
                    # Cluster and evaluate
                    labels = self._fit_hdbscan(reduced_data, h_params)
                    metrics = self._evaluate_clustering(reduced_data, labels)
                    
                    # Store results
                    results.append({
                        **u_params,
                        **h_params,
                        **metrics
                    })
                except Exception as e:
                    logging.warning(f"Failed for params: {u_params} {h_params}: {str(e)}")
        
        results_df = pd.DataFrame(results)
        best_params = self._select_best_parameters(results_df)
        
        return best_params, results_df

    def _fit_umap(self, data: np.ndarray, params: Dict) -> np.ndarray:
        """Fit UMAP with given parameters"""
        if self.config.umap_parametric:
            umap = ParametricUMAP(
                n_components=params['umap_n_components'],
                n_neighbors=params['umap_n_neighbors'],
                min_dist=params['umap_min_dist'],
                metric=self.config.umap_metric
            )
        else:
            umap = UMAP(
                n_components=params['umap_n_components'],
                n_neighbors=params['umap_n_neighbors'],
                min_dist=params['umap_min_dist'],
                metric=self.config.umap_metric
            )
        
        self.umap_model = umap
        return umap.fit_transform(data)

    def _fit_hdbscan(self, data: np.ndarray, params: Dict) -> np.ndarray:
        """Fit HDBSCAN with given parameters"""
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(params['min_cluster_size']),
            min_samples=int(params['min_samples']),
            cluster_selection_epsilon=float(params['cluster_selection_epsilon']),
            cluster_selection_method=str(params['cluster_selection_method']),
            gen_min_span_tree=True,
            allow_single_cluster=False,
            prediction_data=True
        )
        
        self.hdbscan_model = clusterer
        return clusterer.fit_predict(data)

    def _evaluate_clustering(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """Evaluate clustering results"""
        metrics = {}
        
        # Calculate silhouette score if more than one cluster
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) > 1:
                metrics['silhouette_score'] = silhouette_score(
                    data[non_noise_mask], 
                    labels[non_noise_mask]
                )
        
        # Calculate noise ratio
        metrics['noise_ratio'] = np.mean(labels == -1)
        
        # Add relative validity from HDBSCAN
        metrics['relative_validity_score'] = self.hdbscan_model.relative_validity_
        
        return metrics

    def _select_best_parameters(self, results: pd.DataFrame) -> Dict:
        """Select best parameters using weighted scores"""
        # Normalize metrics
        for metric in self.config.score_weights.keys():
            if metric in results.columns:
                results[f'norm_{metric}'] = (results[metric] - results[metric].min()) / (
                    results[metric].max() - results[metric].min()
                )
        
        # Calculate weighted score
        results['weighted_score'] = sum(
            results[f'norm_{metric}'] * weight 
            for metric, weight in self.config.score_weights.items()
            if f'norm_{metric}' in results.columns
        )
        
        # Get best parameters
        best_row = results.loc[results['weighted_score'].idxmax()]
        return {
            'umap_params': {
                'umap_n_neighbors': best_row['umap_n_neighbors'],
                'umap_n_components': best_row['umap_n_components'],
                'umap_min_dist': best_row['umap_min_dist']
            },
            'hdbscan_params': {
                'min_cluster_size': best_row['min_cluster_size'],
                'min_samples': best_row['min_samples'],
                'cluster_selection_epsilon': best_row['cluster_selection_epsilon'],
                'cluster_selection_method': best_row['cluster_selection_method']
            }
        }
    
    def _generate_umap_params(self) -> List[Dict]:
        """Generate UMAP parameter combinations"""
        params = []
        for n_neighbors in self.config.umap_n_neighbors:
            for n_components in self.config.umap_n_components:
                for min_dist in self.config.umap_min_dist:
                    params.append({
                        'umap_n_neighbors': n_neighbors,
                        'umap_n_components': n_components,
                        'umap_min_dist': min_dist
                    })
        return params
    
    def _generate_hdbscan_params(self) -> List[Dict]:
        """Generate HDBSCAN parameter combinations"""
        params = []
        for min_cluster_size in self.config.min_cluster_sizes:
            for min_samples in self.config.min_samples:
                for cluster_selection_epsilon in self.config.cluster_selection_epsilon:
                    for cluster_selection_method in self.config.cluster_selection_method:
                        params.append({
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'cluster_selection_epsilon': cluster_selection_epsilon,
                            'cluster_selection_method': cluster_selection_method
                        })
        return params
    
    def _get_best_params(self) -> Dict:
        """Get best parameters from hyperparameter search"""
        return self.best_params
    
    def _get_evaluation_results(self) -> pd.DataFrame:
        """Get evaluation results from hyperparameter search"""
        return self.evaluation_results
    
    def _get_umap_model(self) -> Union[UMAP, ParametricUMAP]:
        """Get the trained UMAP model"""
        return self.umap_model
    
    def _get_hdbscan_model(self) -> hdbscan.HDBSCAN:
        """Get the trained HDBSCAN model"""
        return self.hdbscan_model
    
    

if __name__ == "__main__":
    import torch
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer, AutoModel
    from src.embedding.clustering import ClusteringConfig, ClusteringPipeline
    from src.embedding.cluster_summarizer import ClusterSummarizer
    from sentence_transformers import SentenceTransformer

    # 1. Load and prepare data
    dataset = pd.read_csv('src/embedding/test_corpus.csv')
    texts = dataset['text'].tolist()
    true_labels = dataset['topic'].tolist()

    # Show first few examples
    print("\nSample texts:")
    for text in texts[:3]:
        print(f"- {text}")
    
    # Show corresp label
    print("\nTrue labels:")
    for label in true_labels[:3]:
        print(f"- {label}")


    # 2. Generate embeddings
    # model = SentenceTransformer('all-mpnet-base-v2')

    MODEL_NAME = "jinaai/jina-embeddings-v3"
    LORA_TASK = "separation"
    print(f"Using model: {MODEL_NAME}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    print("Tokenizer loaded successfully")

    print("Loading base model...")
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, device_map='mps', torch_dtype=torch.float16
    )

    print("Encoding texts...")
    embeddings = model.encode(texts, task=LORA_TASK, show_progress_bar=True)

    # 3. Configure clustering
    config = ClusteringConfig(
        # UMAP parameters
        umap_n_neighbors=[10, 15, 20],
        umap_n_components=[20,],
        umap_min_dist=[0.0],
        umap_parametric=True,
        
        # HDBSCAN parameters
        min_cluster_sizes=[5, 10],
        min_samples=[5, 10, 15],
        cluster_selection_epsilon=[0.01],
        cluster_selection_method=['eom', 'leaf'],
        
        # Scoring weights
        score_weights={
            'silhouette_score': 2.0,
            'noise_ratio': -1.5,
            'relative_validity_score': 1.0
        }
    )

    # 4. Initialize and run clustering pipeline
    pipeline = ClusteringPipeline(config)
    cluster_labels, evaluation_results = pipeline.fit_transform(embeddings)

    # 5. Create final dataset with clusters
    results_df = pd.DataFrame({
        'text': texts,
        'cluster': cluster_labels
    })

    # 6. Analyze results
    print("\nClustering Results:")
    print(f"Number of clusters: {len(np.unique(cluster_labels[cluster_labels != -1]))}")
    print(f"Noise points: {sum(cluster_labels == -1)}")

    # Print best parameters
    print("\nBest Parameters:")
    print(f"UMAP: {pipeline.best_params['umap_params']}")
    print(f"HDBSCAN: {pipeline.best_params['hdbscan_params']}")

    # Show evaluation metrics
    print(evaluation_results[['silhouette_score', 'noise_ratio', 'relative_validity_score']].describe())

    # Calculate ARI wrt true labels
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f"\nAdjusted Rand Index (ARI) with true labels: {ari}")

    # Summarize clusters
    summarizer = ClusterSummarizer(model="gpt-4o-2024-05-13", batch_size=20)
    # Get summaries with error handling
    summaries, parsed, formatted = summarizer.summarize_clusters(
        results_df,
        text_column='text',
        cluster_column='cluster',
        max_samples=100
    )

    # Show summaries
    print(f"SUMMARY KEYS:")
    print(f"{summaries.keys()}")

    # Print summaries
    print("\nCluster Summaries:")
    summarizer.print_summaries(formatted)

    # Save pipeline and summaries
    pipeline.save('artifacts/clustering/models', parsed)
    # summarizer.save('cluster_summaries')

    # Load pipeline and summaries
    pipeline, parsed = ClusteringPipeline.load('artifacts/clustering/models')
    # summarizer = ClusterSummarizer.load('cluster_summaries')

    # # Show clusters
    # for cluster in np.unique(cluster_labels):
    #     if cluster == -1:
    #         continue
    #     cluster_texts = results_df[results_df['cluster'] == cluster]['text'].tolist()
    #     print(f"\nCluster {cluster}:")
    #     for text in cluster_texts[:3]:  # Show first 3 examples
    #         print(f"- {text}")