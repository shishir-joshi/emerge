import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

@dataclass
class ChunkTopicMatch:
    """Detailed analysis of chunk-topic match patterns"""
    chunk_text: str
    chunk_tokens: List[str]
    description_similarity: np.ndarray
    example_similarities: List[Dict[str, any]]
    max_score: float
    avg_example_score: float


@dataclass
class ChunkAnalysisResult:
    """Analysis result for a chunk including all similarity matrices"""
    chunk_text: str
    span_indices: Tuple[int, int]
    topic_matches: Dict[int, Dict[str, any]]  # topic_id -> {description_matrix, example_matrices, scores}
    

@dataclass
class ChunkMetrics:
    """Metrics for chunk selection"""
    chunk_text: str
    max_topic_score: float
    score_variance: float
    desc_example_gap: float
    topic_confusion_score: float  # How ambiguous between topics
    length: int

class ChunkSelector:
    """Selects most informative chunks for analysis"""
    
    @staticmethod
    def compute_chunk_metrics(chunk_text: str, topic_matches: Dict) -> ChunkMetrics:
        """Compute metrics for chunk selection"""
        topic_scores = []
        desc_example_gaps = []
        
        for topic_id, match_data in topic_matches.items():
            desc_score = match_data['description_score']
            max_example_score = max(match_data['example_scores']) if match_data['example_scores'] else 0
            topic_scores.append(desc_score)
            desc_example_gaps.append(abs(desc_score - max_example_score))
        
        return ChunkMetrics(
            chunk_text=chunk_text,
            max_topic_score=max(topic_scores),
            score_variance=np.var(topic_scores),
            desc_example_gap=np.mean(desc_example_gaps),
            topic_confusion_score=1 - (max(topic_scores) - np.mean(topic_scores)),
            length=len(chunk_text.split())
        )
    
    @staticmethod
    def select_representative_chunks(analyses: Dict[str, ChunkAnalysisResult], 
                                  max_chunks: int = 50) -> List[str]:
        """Select most informative chunks for visualization"""
        
        # Compute metrics for all chunks
        chunk_metrics = []
        for chunk_text, analysis in analyses.items():
            metrics = ChunkSelector.compute_chunk_metrics(
                chunk_text, 
                analysis.topic_matches
            )
            chunk_metrics.append(metrics)
        
        # Select diverse set of chunks
        selected_chunks = []
        
        # 1. High impact chunks (strong matches)
        high_impact = sorted(
            chunk_metrics,
            key=lambda x: x.max_topic_score,
            reverse=True
        )[:max_chunks // 4]
        selected_chunks.extend(m.chunk_text for m in high_impact)
        
        # 2. Ambiguous chunks (high topic confusion)
        ambiguous = sorted(
            [m for m in chunk_metrics if m.chunk_text not in selected_chunks],
            key=lambda x: x.topic_confusion_score,
            reverse=True
        )[:max_chunks // 4]
        selected_chunks.extend(m.chunk_text for m in ambiguous)
        
        # 3. Chunks with high desc-example divergence
        divergent = sorted(
            [m for m in chunk_metrics if m.chunk_text not in selected_chunks],
            key=lambda x: x.desc_example_gap,
            reverse=True
        )[:max_chunks // 4]
        selected_chunks.extend(m.chunk_text for m in divergent)
        
        # 4. Edge cases (very short/long chunks)
        length_sorted = sorted(
            [m for m in chunk_metrics if m.chunk_text not in selected_chunks],
            key=lambda x: abs(x.length - np.mean([m.length for m in chunk_metrics]))
        )[:max_chunks - len(selected_chunks)]
        selected_chunks.extend(m.chunk_text for m in length_sorted)
        
        return selected_chunks

class SimilarityDashboard:
    """Simplified dashboard for analyzing topic-chunk matches"""
    
    def plot_topic_chunk_heatmap(self, analyses: Dict[str, ChunkAnalysisResult], 
                                analyzer_instance,
                                matched_topic_ids: List[int]):
        """Plot overview heatmap of topic-chunk matches"""
        # Prepare data
        chunks = list(analyses.keys())
        topics = [analyzer_instance.topics[tid].name for tid in matched_topic_ids]
        
        # Create score matrix
        scores = np.zeros((len(topics), len(chunks)))
        for i, topic_id in enumerate(matched_topic_ids):
            for j, (chunk, analysis) in enumerate(analyses.items()):
                if topic_id in analysis.topic_matches:
                    scores[i,j] = analysis.topic_matches[topic_id]['description_score']
        
        # Plot heatmap
        plt.figure(figsize=(15,8))
        sns.heatmap(scores, 
                   xticklabels=[c[:30] + '...' for c in chunks],
                   yticklabels=topics,
                   cmap='RdBu_r',
                   center=0.5)
        plt.title('Topic-Chunk Similarity Overview')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(str(self.output_dir / 'topic_chunk_overview.png'))
        plt.close()

    def plot_score_distributions(self, analyses: Dict[str, ChunkAnalysisResult],
                               analyzer_instance,
                               matched_topic_ids: List[int]):
        """Plot distribution of similarity scores per topic"""
        # Collect scores
        topic_scores = defaultdict(list)
        for analysis in analyses.values():
            for topic_id in matched_topic_ids:
                if topic_id in analysis.topic_matches:
                    match_data = analysis.topic_matches[topic_id]
                    topic_scores[analyzer_instance.topics[topic_id].name].append({
                        'description': match_data['description_score'],
                        'example': max(match_data['example_scores']) if match_data['example_scores'] else 0
                    })
        
        # Plot distributions
        fig, ax = plt.subplots(figsize=(12,6))
        positions = np.arange(len(matched_topic_ids)) * 3
        width = 0.35
        
        for i, topic_id in enumerate(matched_topic_ids):
            topic_name = analyzer_instance.topics[topic_id].name
            scores = topic_scores[topic_name]
            
            desc_scores = [s['description'] for s in scores]
            example_scores = [s['example'] for s in scores]
            
            ax.boxplot([desc_scores, example_scores], 
                      positions=[positions[i], positions[i]+width],
                      labels=['Desc', 'Ex'])
        
        ax.set_xticks(positions + width/2)
        ax.set_xticklabels([t[:20] for t in topics], rotation=45, ha='right')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Score Distributions by Topic')
        plt.tight_layout()
        plt.savefig(str(self.output_dir / 'score_distributions.png'))
        plt.close()

    def plot_confusion_analysis(self, analyses: Dict[str, ChunkAnalysisResult]):
        """Plot analysis of topic confusion"""
        chunk_metrics = []
        for chunk_text, analysis in analyses.items():
            metrics = ChunkSelector.compute_chunk_metrics(chunk_text, analysis.topic_matches)
            chunk_metrics.append(metrics)
        
        # Create scatter plot
        plt.figure(figsize=(10,6))
        plt.scatter([m.topic_confusion_score for m in chunk_metrics],
                   [m.max_topic_score for m in chunk_metrics],
                   alpha=0.6)
        
        # Add labels for interesting points
        for metric in chunk_metrics:
            if metric.topic_confusion_score > 0.7 or metric.max_topic_score > 0.7:
                plt.annotate(metric.chunk_text[:30],
                           (metric.topic_confusion_score, metric.max_topic_score),
                           xytext=(5,5), textcoords='offset points',
                           fontsize=8)
        
        plt.xlabel('Topic Confusion Score')
        plt.ylabel('Max Topic Score')
        plt.title('Topic Confusion Analysis')
        plt.tight_layout()
        plt.savefig(str(self.output_dir / 'confusion_analysis.png'))
        plt.close()

    def create_dashboard(self, analyzer_instance, sentence: str,
                        analyses: Dict[str, ChunkAnalysisResult],
                        matched_topic_ids: List[int]):
        """Generate all visualizations"""
        self.plot_topic_chunk_heatmap(analyses, analyzer_instance, matched_topic_ids)
        self.plot_score_distributions(analyses, analyzer_instance, matched_topic_ids)
        self.plot_confusion_analysis(analyses)

    def __init__(self, output_dir: str = "analysis_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_matches(self, 
                       analyzer_instance,
                       sent_id: str,
                       matched_topic_ids: List[int]) -> Dict[str, ChunkAnalysisResult]:
        """Extract all similarity matrices and scores from cached prediction"""
        
        cache = analyzer_instance.sentences[sent_id]
        sentence = cache['sentence']
        chunks = cache['chunks']
        
        # Get document embeddings using late chunking
        outputs, offset_mapping = analyzer_instance.embed_full_document(sentence)
        chunk_tokens_list, span_indices = analyzer_instance.get_late_chunked_embeddings(
            outputs, sentence, offset_mapping, chunks
        )
        
        # Get the actual tokens from the tokenizer
        input_ids = analyzer_instance.tokenizer(sentence)['input_ids']  # First batch
        all_tokens = analyzer_instance.tokenizer.convert_ids_to_tokens(input_ids)
        
        chunk_analyses = {}
        
        # Analyze each chunk
        for chunk_idx, (chunk, chunk_tokens) in enumerate(zip(chunks, chunk_tokens_list)):
            start_idx, end_idx = span_indices[chunk_idx]
            chunk_token_texts = all_tokens[start_idx:end_idx]  # Get actual tokens for this span
            
            topic_matches = {}
            
            # Get matches for each topic
            for topic_id in matched_topic_ids:
                topic_examples = analyzer_instance.topic_token_embeddings[topic_id]
                desc_example = next(ex for ex in topic_examples if ex['is_description'])
                other_examples = [ex for ex in topic_examples if not ex['is_description']]

                # Get topic tokens directly from the example
                topic_token_texts = analyzer_instance.tokenizer.convert_ids_to_tokens(
                    analyzer_instance.tokenizer(desc_example['text'])['input_ids']
                )
                
                # Get description similarity
                with torch.no_grad():
                    desc_matrix = torch.matmul(
                        desc_example['token_embeddings'], 
                        chunk_tokens.transpose(0, 1)
                    ).cpu().numpy()
                
                # Get example similarities
                example_matrices = []
                for ex in other_examples:
                    with torch.no_grad():
                        ex_matrix = torch.matmul(
                            ex['token_embeddings'],
                            chunk_tokens.transpose(0, 1)
                        ).cpu().numpy()
                        example_matrices.append({
                            'text': ex['text'],
                            'matrix': ex_matrix,
                            'tokens': analyzer_instance.tokenizer.convert_ids_to_tokens(
                                # ex['input_ids'][0]  # First batch
                                analyzer_instance.tokenizer(ex['text'])['input_ids']
                            )
                        })
                
                topic_matches[topic_id] = {
                    'description_matrix': desc_matrix,
                    'example_matrices': example_matrices,
                    'description_score': desc_matrix.max(),
                    'example_scores': [ex['matrix'].max() for ex in example_matrices],
                    'topic_tokens': topic_token_texts,
                    'chunk_tokens': chunk_token_texts
                }
            
            chunk_analyses[chunk] = ChunkAnalysisResult(
                chunk_text=chunk,
                span_indices=span_indices[chunk_idx],
                topic_matches=topic_matches
            )
        
        return chunk_analyses
            

if __name__ == "__main__":
    from src.curation.analyzers.similarity_dashboard import SimilarityDashboard

    # Initialize dashboard
    dashboard = SimilarityDashboard()

    # Analyze matches
    analyses = dashboard.analyze_matches(
        analyzer_instance=analyzer,
        sent_id="sample_turn_1",
        matched_topic_ids=matched_topic_ids
    )

    # Create and display dashboard
    dashboard.create_dashboard(
        analyzer_instance=analyzer,
        sentence=analyzer.sentences["sample_turn_1"]["sentence"],
        analyses=analyses,
        matched_topic_ids=matched_topic_ids,
        min_similarity=0.3
    )