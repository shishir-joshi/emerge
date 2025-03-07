import json
import torch
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from itables import show
from pathlib import Path
from datetime import datetime
from IPython.display import display, HTML, Javascript
from dataclasses import dataclass, field
from src.embedding.clustering import ClusteringConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalyzerConfig:
    """Configuration for data curation"""
    min_len: int = 5
    max_len: int = 400
    stride: int = 1
    batch_size: int = 32
    top_k: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    clustering_config: ClusteringConfig = field(default_factory=ClusteringConfig)
    base_color: tuple = (102, 217, 239)


class BaseAnalyzer(ABC):
    def __init__(self, config: AnalyzerConfig, cache_save_dir=None):
        self.config = config
        self.cache_save_dir = Path(cache_save_dir) if cache_save_dir else None
        self.sentences = {}
        self.logger = logging.getLogger(__name__)

    # Core Abstract Methods
    @abstractmethod
    def embed_chunks(self, chunks):
        """Embed chunks of text."""
        pass

    @abstractmethod
    def predict_clusters(self, chunks, chunks_embs, verbose=False):
        """Predict clusters for chunks."""
        pass

    # Span Generation
    def create_overlapping_chunks_with_stride(self, sentence, stride=1):
        words = np.array(sentence.split())
        n = len(words)
        chunks = []

        indices = np.arange(0, n - self.config.min_len + 1, stride)
        for i in indices:
            lengths = np.arange(self.config.min_len, min(self.config.max_len, n - i) + 1)
            end_indices = i + lengths
            chunks.extend([" ".join(words[i:end]) for end in end_indices])

        return chunks

    # Prediction Pipeline
    def predict(self, sentence, stride, min_len, max_len, sent_id, cache_sent=False, verbose=True, top_k=10):
        logging.info("---------------------------------------------------")
        if sent_id not in self.sentences:
            logging.info(f"Predicting clusters for sentence ID {sent_id}.")
            self.config.min_len = min_len
            self.config.max_len = max_len

            logging.info(f"Creating overlapping chunks with stride {stride}.")
            # Step 1: Create overlapping chunks with custom stride
            chunks = self.create_overlapping_chunks_with_stride(sentence, stride)

            logging.info(f"Filtering chunks by length.")
            # Step 2: Filter chunks by length
            filtered_chunks = [chunk for chunk in chunks if min_len <= len(chunk.split()) <= max_len]

            logging.info(f"Embedding filtered chunks.")
            # Step 3: Embed the filtered chunks
            chunk_embeddings = self.embed_chunks(filtered_chunks)

            logging.info(f"Predicting clusters for filtered chunks.")
            # Step 4: Predict clusters for each chunk
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

        logging.info(f"Filtering out noise clusters.")
        # Step 5: Filter out noise clusters
        cluster_predictions = cluster_predictions[cluster_predictions[self.config.clustering_config.cluster_labels_column] != -1]

        logging.info(f"Identifying top {top_k} Pareto optimal clusters.")
        if min_len is not None:
            cluster_predictions = cluster_predictions[cluster_predictions['TURN_LEN'] >= min_len]
        if max_len is not None:
            cluster_predictions = cluster_predictions[cluster_predictions['TURN_LEN'] <= max_len]
            
        pareto_cluster_ids, score_df, scores = self._get_pareto_optimal_clusters(
            cluster_predictions,
            sentence,
            top_k=top_k
        )

        logging.info(f"Getting the most representative span (longest span) from the top cluster.")
        # Step 6: Get the most representative spans for each Pareto optimal cluster
        representative_spans = []
        for cluster_id in pareto_cluster_ids:
            cluster_df = cluster_predictions[cluster_predictions[self.config.clustering_config.cluster_labels_column] == cluster_id]
            if not cluster_df.empty:
                max_len_span = cluster_df.loc[cluster_df['TURN_LEN'].idxmax()]
                representative_spans.append(max_len_span)
        
        representative_span = pd.DataFrame(representative_spans)
        logging.info("---------------------------------------------------")
        return representative_span.to_frame().T, pareto_cluster_ids, score_df, scores

    # Span Operations
    def _merge_overlapping_spans(self, spans):
        if not spans:
            return 0
        
        merged = []
        spans.sort()
        current_start, current_end = spans[0]
        
        for start, end in spans[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        return sum(end - start for start, end in merged)

    # Pareto Optimization
    def _get_pareto_optimal_clusters(self, cluster_predictions, sentence, top_k=10):
        """Get Pareto optimal clusters using both structural and semantic features."""
        metrics = {}
        sentence_len = len(sentence)
        
        # 1. Calculate core metrics
        for cluster_id, cluster_df in cluster_predictions.groupby(self.config.clustering_config.cluster_labels_column):
            if cluster_id == -1:
                continue
                
            # Calculate span coverage
            spans = []
            for text in cluster_df[self.config.clustering_config.text_column]:
                start_idx = sentence.find(text)
                if start_idx != -1:
                    spans.append((start_idx, start_idx + len(text)))
            
            total_coverage = self._merge_overlapping_spans(spans)
            
            # Add semantic score metrics if they exist
            avg_semantic_score = 0.0
            max_semantic_score = 0.0
            if 'relevance_score' in cluster_df.columns:
                avg_semantic_score = cluster_df['relevance_score'].mean()
                max_semantic_score = cluster_df['relevance_score'].max()
            
            metrics[cluster_id] = {
                'frequency': len(cluster_df),                    # Number of occurrences
                'coverage_ratio': total_coverage / sentence_len, # Normalized coverage
                # 'avg_length': cluster_df['TURN_LEN'].mean(),     # Average span length
                'avg_semantic_score': avg_semantic_score,        # Average semantic score
                'max_semantic_score': max_semantic_score         # Max semantic score
            }
        
        # 2. Convert to numpy array for vectorized operations
        metrics_df = pd.DataFrame(metrics).T
        cluster_ids = metrics_df.index.values
        values = metrics_df.values  # Shape: (n_points, n_dimensions)

        # 3. Vectorized dominance check
        # Broadcasting: (n_points, 1, n_dimensions) >= (1, n_points, n_dimensions)
        dominance = (values[:, np.newaxis, :] >= values[np.newaxis, :, :])
        strict_dominance = (values[:, np.newaxis, :] > values[np.newaxis, :, :])
        
        # Point i is dominated if there exists a point j where:
        # - j dominates i in all dimensions (all True in dominance[j,i])
        # - j strictly dominates i in at least one dimension (any True in strict_dominance[j,i])
        dominated = np.any(
            (dominance.all(axis=2)) & (strict_dominance.any(axis=2)), 
            axis=0
        )

        # 4. Get non-dominated points (Pareto front)
        pareto_mask = ~dominated
        pareto_front = cluster_ids[pareto_mask]
        
        if len(pareto_front) == 0:
            return [], metrics_df, []

        # 5. Handle scoring based on number of Pareto optimal points
        pareto_values = values[pareto_mask]
        
        if len(pareto_values) == 1:
            # For single cluster, use score of 1.0
            scores = np.array([1.0])
            top_clusters = pareto_front
            top_scores = scores.tolist()
        else:
            # For multiple clusters, normalize and rank
            value_ranges = pareto_values.max(axis=0) - pareto_values.min(axis=0)
            valid_dims = value_ranges > 0
            
            if not valid_dims.any():
                # All dimensions have same values, treat clusters as equal
                scores = np.ones(len(pareto_values)) / len(pareto_values)
            else:
                # Normalize only valid dimensions
                normalized_values = np.zeros_like(pareto_values, dtype=np.float32)
                normalized_values[:, valid_dims] = (
                    (pareto_values[:, valid_dims] - pareto_values.min(axis=0)[valid_dims]) / 
                    value_ranges[valid_dims]
                )
                scores = normalized_values.sum(axis=1) / valid_dims.sum()
            
            # Get top-k results
            top_k = min(top_k, len(scores))  # Ensure we don't exceed array bounds
            top_k_indices = np.argsort(-scores)[:top_k]
            top_clusters = pareto_front[top_k_indices]
            top_scores = scores[top_k_indices].tolist()
        
        logger.debug(f"Pareto front size: {len(pareto_front)}")
        logger.debug(f"Final scores: {top_scores}")
        
        return top_clusters.tolist(), metrics_df, top_scores

    # Sentence Management
    def get_sentence(self, sent_id: str) -> dict:
        if sent_id in self.sentences:
            return self.sentences[sent_id]
            
        if self.cache_save_dir and self.cache_save_dir.exists():
            try:
                if not hasattr(self, 'sentence_mappings'):
                    with open(self.cache_save_dir / "sentence_mappings.json", "r") as f:
                        self.sentence_mappings = json.load(f)
                
                if sent_id in self.sentence_mappings:
                    mapping = self.sentence_mappings[sent_id]
                    
                    emb_data = np.load(self.cache_save_dir / mapping["embedding_file"])
                    predicted_df = pd.read_parquet(self.cache_save_dir / mapping["predictions_file"])
                    
                    self.sentences[sent_id] = {
                        "sentence": mapping["sentence"],
                        "chunks": mapping["chunks"],
                        "chunks_embs": emb_data["chunks_embs"],
                        "predicted_df": predicted_df
                    }
                    return self.sentences[sent_id]
                    
            except Exception as e:
                logging.error(f"Error loading sentence {sent_id} from disk: {e}")
                
        raise KeyError(f"Sentence ID {sent_id} not found in memory or on disk")

    def __getitem__(self, sent_id: str) -> dict:
        return self.get_sentence(sent_id)

    # Cache Management
    def save_cache(self, save_dir: str) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        mapping_file = save_path / "sentence_mappings.json"
        if mapping_file.exists():
            with open(mapping_file, "r") as f:
                sentence_mappings = json.load(f)
        else:
            sentence_mappings = {}
        
        metadata = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "num_sentences": len(self.sentences)
        }
        
        saved_count = 0
        skipped_count = 0
        
        for sent_id, content in self.sentences.items():
            if sent_id in sentence_mappings:
                logging.debug(f"Skipping {sent_id} - already saved")
                skipped_count += 1
                continue
            
            sentence_mappings[sent_id] = {
                "sentence": content["sentence"],
                "chunks": content["chunks"],
                "embedding_file": f"emb_{sent_id}.npz",
                "predictions_file": f"pred_{sent_id}.parquet"
            }
            
            np.savez_compressed(
                save_path / f"emb_{sent_id}.npz",
                chunks_embs=content["chunks_embs"]
            )
            
            content["predicted_df"].to_parquet(
                save_path / f"pred_{sent_id}.parquet",
                compression="snappy"
            )
            
            saved_count += 1
        
        with open(mapping_file, "w") as f:
            json.dump(sentence_mappings, f, indent=2)
            
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logging.info(f"Saved {saved_count} new sentences, skipped {skipped_count} existing")

    def load_cache(self, save_dir: str) -> None:
        save_path = Path(save_dir)
        if not save_path.exists():
            raise FileNotFoundError(f"Save directory {save_path} not found")
        
        with open(save_path / "sentence_mappings.json", "r") as f:
            sentence_mappings = json.load(f)
        
        self.sentences = {}
        
        for sent_id, mapping in sentence_mappings.items():
            emb_data = np.load(save_path / mapping["embedding_file"])
            chunks_embs = emb_data["chunks_embs"]
            
            predicted_df = pd.read_parquet(save_path / mapping["predictions_file"])
            
            self.sentences[sent_id] = {
                "sentence": mapping["sentence"],
                "chunks": mapping["chunks"],
                "chunks_embs": chunks_embs,
                "predicted_df": predicted_df
            }
        
        logging.info(f"Successfully loaded analyzer state from {save_path}")

    # Visualization
    def highlight_with_attention(self, sentence, chunks, clusters=None, titles=None):
        """
        Highlight sentence with attention based on chunks and clusters.
        
        Args:
            sentence: Text to highlight
            chunks: List of text spans
            clusters: List of cluster IDs for each chunk
            titles: List of cluster titles
        """

        # 1. Build the span index
        def build_span_index(sentence, chunks, clusters):
            """Create index of character positions to their cluster IDs"""
            span_index = {}
            for chunk, cluster in zip(chunks, clusters or [None] * len(chunks)):
                start = sentence.find(chunk)
                if start != -1:
                    for i in range(start, start + len(chunk)):
                        if i not in span_index:
                            span_index[i] = set()
                        span_index[i].add(cluster)
            return span_index

        # 2. Setup color mapping
        # def setup_colors(clusters):
        #     """Setup color scheme for clusters"""
        #     if clusters:
        #         unique_clusters = sorted(set(clusters))
        #         color_map = cm.get_cmap('tab20', len(unique_clusters))
        #         return {c: mcolors.rgb2hex(color_map(i)[:3]) 
        #             for i, c in enumerate(unique_clusters)}
        #     return {None: f"rgba({self.config.base_color[0]}, {self.config.base_color[1]}, {self.config.base_color[2]}, 0.5)"}
        
        def setup_colors(clusters):
            """Setup advanced color scheme with gradients for better dark theme visibility"""
            if not clusters:
                return {None: "linear-gradient(135deg, rgba(102, 217, 239, 0.5), rgba(166, 226, 246, 0.5))"}
            
            # Color palettes inspired by cyberpunk and synthwave aesthetics
            base_colors = [
                ("FF00FF", "00FFFF"),  # Magenta to Cyan
                ("00FF00", "FFFF00"),  # Green to Yellow
                ("FF8800", "FF0088"),  # Orange to Pink
                ("00FFFF", "0088FF"),  # Cyan to Blue
                ("FFFF00", "FF8800"),  # Yellow to Orange
                ("FF0088", "8800FF"),  # Pink to Purple
                ("0088FF", "00FF88"),  # Blue to Mint
                ("88FF00", "FFFF00"),  # Lime to Yellow
                ("FF0000", "FF88FF"),  # Red to Light Pink
                ("00FF88", "88FFFF"),  # Mint to Light Cyan
            ]
            
            unique_clusters = sorted(set(clusters))
            color_map = {}
            
            for i, cluster in enumerate(unique_clusters):
                start, end = base_colors[i % len(base_colors)]
                # Create gradient background
                gradient = f"linear-gradient(135deg, #{start}60, #{end}60)"
                color_map[cluster] = gradient
                
            return color_map

        # Build data structures
        span_index = build_span_index(sentence, chunks, clusters)
        cluster_colors = setup_colors(clusters)
        cluster_titles = dict(zip(clusters, titles)) if clusters and titles else {}
        container_id = f"highlight-container-{hash(sentence)}"

        # 3. Generate HTML
        def generate_sentence_html():
            html = []
            for i, char in enumerate(sentence):
                if i in span_index:
                    main_cluster = next(iter(span_index[i]))
                    background = cluster_colors[main_cluster]
                    tooltip = cluster_titles.get(main_cluster, f"Cluster {main_cluster}")
                    
                    all_clusters = ' '.join(f'cluster-{c}' for c in span_index[i])
                    cluster_ids = ' '.join(str(c) for c in span_index[i])
                    
                    html.append(
                        f'<span class="highlight {all_clusters}" '
                        f'data-cluster-ids="{cluster_ids}" '
                        f'style="background: {background};" '
                        f'data-tooltip="{tooltip}">{char}</span>'
                    )
                else:
                    html.append(char)
            return ''.join(html)

        def generate_legend_html():
            """Generate HTML for the legend"""
            items = []
            for cluster in cluster_colors:
                title = cluster_titles.get(cluster, '')
                tooltip = f"{title}" if title else f"Cluster {cluster}"
                gradient = cluster_colors[cluster]
                items.append(f'''
                    <div class="legend-item cluster-{cluster}" 
                        data-cluster-id="{cluster}"
                        style="display: flex; align-items: center; padding: 5px; cursor: pointer;">
                        <div class="legend-color" style="width: 20px; height: 20px; background: {gradient}; 
                            margin-right: 5px; border-radius: 3px;"></div>
                        <span>{tooltip}</span>
                    </div>
                ''')
            return f"<div id='legend' style='margin-top: 20px; display: flex; flex-wrap: wrap; gap: 10px;'>\n{''.join(items)}\n</div>"

        # Combine HTML components
        html_content = f"""
        <div id="{container_id}" style="font-size:18px; font-family:Arial, sans-serif;">
            <style>
                #{container_id} .highlight {{
                    padding: 2px;
                    border-radius: 3px;
                    position: relative;
                    transition: all 0.3s;
                    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                #{container_id} .highlight.dim {{
                    opacity: 0.15;
                    text-shadow: none;
                    box-shadow: none;
                }}
                #{container_id} .highlight {{
                    padding: 2px;
                    border-radius: 3px;
                    position: relative;
                    transition: opacity 0.3s;
                }}
                #{container_id} .highlight.dim {{
                    opacity: 0.2;
                }}
                #{container_id} .legend-item {{
                    transition: opacity 0.3s;
                }}
                #{container_id} .legend-item.dim {{
                    opacity: 0.2;
                }}
            </style>
            <div class="sentence">{generate_sentence_html()}</div>
            {generate_legend_html()}
        </div>
        """

        # JavaScript for interaction
        js_code = f"""
        (() => {{
            const container = document.getElementById('{container_id}');
            if (!container) return;

            const highlights = container.getElementsByClassName('highlight');
            const legendItems = container.getElementsByClassName('legend-item');

            function highlightClusterSpans(clusterId) {{
                // Dim all spans and legend items first
                Array.from(highlights).forEach(h => h.classList.add('dim'));
                Array.from(legendItems).forEach(l => l.classList.add('dim'));

                // Un-dim spans containing the cluster ID
                Array.from(highlights).forEach(h => {{
                    const clusterIds = h.dataset.clusterIds.split(' ');
                    if (clusterIds.includes(clusterId)) {{
                        h.classList.remove('dim');
                    }}
                }});

                // Un-dim matching legend item
                Array.from(legendItems).forEach(l => {{
                    if (l.dataset.clusterId === clusterId) {{
                        l.classList.remove('dim');
                    }}
                }});
            }}

            function resetHighlights() {{
                Array.from(highlights).forEach(h => h.classList.remove('dim'));
                Array.from(legendItems).forEach(l => l.classList.remove('dim'));
            }}

            // Add hover handlers
            Array.from(highlights).forEach(h => {{
                h.addEventListener('mouseenter', (e) => {{
                    const clusterIds = e.target.dataset.clusterIds.split(' ');
                    if (clusterIds.length > 0) {{
                        highlightClusterSpans(clusterIds[0]);
                    }}
                }});
                h.addEventListener('mouseleave', resetHighlights);
            }});

            Array.from(legendItems).forEach(l => {{
                l.addEventListener('mouseenter', (e) => {{
                    const clusterId = e.target.dataset.clusterId;
                    if (clusterId) {{
                        highlightClusterSpans(clusterId);
                    }}
                }});
                l.addEventListener('mouseleave', resetHighlights);
            }});
        }})();
        """

        display(HTML(html_content))
        display(Javascript(js_code))
        return None
    
    def analyze(self, sentence_id, pareto_cluster_ids, min_len=None, max_len=None):
        """
        Analyze sentence and highlight spans based on Pareto optimal clusters.
        
        Args:
            sentence_id: ID of sentence to analyze
            pareto_cluster_ids: List of cluster IDs ordered by Pareto optimality
            min_len: Minimum length filter
            max_len: Maximum length filter
        """
        if sentence_id not in self.sentences:
            raise ValueError(f"No sentence found with ID: {sentence_id}")

        sentence_data = self.sentences[sentence_id]
        sentence = sentence_data['sentence']
        predicted_df = sentence_data['predicted_df']

        # Apply length filters if specified
        if min_len is not None:
            predicted_df = predicted_df[predicted_df['TURN_LEN'] >= min_len]
        if max_len is not None:
            predicted_df = predicted_df[predicted_df['TURN_LEN'] <= max_len]

        # Filter noise and order by pareto_cluster_ids
        filtered_df = predicted_df[predicted_df[self.config.clustering_config.cluster_labels_column] != -1]
        
        # 1. Split into pareto and non-pareto
        pareto_df = filtered_df[filtered_df[self.config.clustering_config.cluster_labels_column].isin(pareto_cluster_ids)]
        non_pareto_df = filtered_df[~filtered_df[self.config.clustering_config.cluster_labels_column].isin(pareto_cluster_ids)]

        # 2. Order pareto part
        ordered_parts = []
        for cluster_id in pareto_cluster_ids:
            cluster_slice = pareto_df[pareto_df[self.config.clustering_config.cluster_labels_column] == cluster_id]
            ordered_parts.append(cluster_slice)

        # 3. Combine ordered pareto + remaining
        ordered_df = pd.concat(ordered_parts + [non_pareto_df])
        
        # 4. Group and show
        summary_df = ordered_df.groupby(
            [self.config.clustering_config.cluster_title_column, self.config.clustering_config.cluster_labels_column]
        )[[self.config.clustering_config.text_column, 'TURN_LEN']].agg({
            self.config.clustering_config.text_column: 'count',
            'TURN_LEN': ['min', 'max']
        })

        show(summary_df, classes='display wrap compact')

        # Get chunks and metadata for highlighting
        selected_data = filtered_df[
            filtered_df[self.config.clustering_config.cluster_labels_column].isin(pareto_cluster_ids)
        ][[self.config.clustering_config.text_column, self.config.clustering_config.cluster_labels_column, self.config.clustering_config.cluster_title_column]]
        
        selected_chunks = selected_data[self.config.clustering_config.text_column].values.tolist()
        selected_clusters = selected_data[self.config.clustering_config.cluster_labels_column].values.tolist()
        selected_titles = selected_data[self.config.clustering_config.cluster_title_column].values.tolist()

        # Highlight with clusters and titles
        highlighted_sentence = self.highlight_with_attention(
            sentence, 
            selected_chunks,
            selected_clusters,
            selected_titles
        )
        display(HTML(f'<p style="font-size:18px; font-family:Arial, sans-serif;">{highlighted_sentence}</p>'))

        return summary_df
