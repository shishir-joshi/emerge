import torch
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass, field
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from collections import defaultdict

from src.curation.analyzers.base_analyzer import BaseAnalyzer, AnalyzerConfig
from src.curation.utils.span_utils import find_chunk_spans
from src.curation.utils.late_chunking import late_chunking

@dataclass
class Topic:
    """Class representing a user-defined topic with optional examples"""
    id: int
    name: str
    description: str
    examples: List[str] = None
    
    @property
    def title(self) -> str:
        return self.name
    
    @property
    def summary(self) -> str:
        return self.description

class LateInteractionCrossEncoder(BaseAnalyzer):
    """
    Analyzer that combines cross-encoder accuracy with late interaction efficiency
    using Jina Embeddings v3 and proper late chunking techniques.
    
    This hybrid approach:
    1. Uses a single pass to generate token-level embeddings for the full document
    2. Performs late interaction scoring similar to ColBERT
    3. Compares spans to pre-computed topic embeddings efficiently
    """
    def __init__(
        self, 
        config: AnalyzerConfig = None,
        topics: List[Topic] = None,
        cache_save_dir: str = None,
        model: AutoModel = None,
        tokenizer: AutoTokenizer = None,
        model_kwargs: dict = {},
        relevance_threshold: float = 0.5,
        prediction_mode: str = "typical",  # late_chunking or typical
    ):
        super().__init__(config, cache_save_dir)
        
        # Set up model if provided
        self.model = model
        self.tokenizer = tokenizer
        self.model_kwargs = model_kwargs
        
        # Initialize model if not provided
        if self.model is None or self.tokenizer is None:
            self.logger.info(f"Loading Jina Embeddings v3 model...")
            self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
            self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
            
        self._setup_model(model_kwargs)
        
        self.relevance_threshold = relevance_threshold
        
        # Initialize topics
        self.topics = {}
        self.topic_token_embeddings = {}
        if topics:
            for topic in topics:
                self.add_topic(topic)
                
        # For compatibility with BaseAnalyzer interface
        self.text_column = 'text'
        self.cluster_column = 'cluster'
        self.cluster_title_column = 'cluster_title'
        self.cluster_summary_column = 'cluster_summary'
        
        # Override clustering config attributes with equivalents if they exist
        if hasattr(self.config, "clustering_config"):
            self.config.clustering_config.text_column = self.text_column
            self.config.clustering_config.cluster_labels_column = self.cluster_column
            self.config.clustering_config.cluster_title_column = self.cluster_title_column
            self.config.clustering_config.cluster_summary_column = self.cluster_summary_column

        self.prediction_mode = prediction_mode
        if prediction_mode not in ["late_chunking", "typical"]:
            raise ValueError("prediction_mode must be either 'late_chunking' or 'typical'")

    def _setup_model(self, model_kwargs):
        """Set up the model with appropriate configuration"""
        self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Set up batch size from kwargs or default
        self.model_batch_size = model_kwargs.get('model_batch_size', 32)
        
        # Handle adapter/LoRA setup if specified
        self.task = model_kwargs.get('lora_task', None)
        if self.task and hasattr(self.model, '_adaptation_map'):
            self.task_id = self.model._adaptation_map[self.task]
            self.adapter_mask = torch.full(
                (1,), self.task_id, dtype=torch.int32
            ).to(self.device)
        else:
            self.adapter_mask = None

    def embed_full_document(self, document):
        """
        Embed a full document once, extracting token-level embeddings.
        Based on the implementation from LateChunkingClusterAnalyzer.
        """
        tokens = self.tokenizer(document, return_tensors='pt', return_offsets_mapping=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        with torch.no_grad():
            if self.adapter_mask is not None:
                outputs = self.model(**tokens, adapter_mask=self.adapter_mask, output_hidden_states=True)
            else:
                outputs = self.model(**tokens, output_hidden_states=True)
                
        return outputs, tokens['offset_mapping'][0].tolist()

    def get_token_embeddings_from_output(self, outputs):
        """Extract token embeddings from model output"""
        # For models that provide hidden_states, use the last layer
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # Get last layer's hidden states (similar to LateChunkingClusterAnalyzer)
            token_embeddings = outputs.hidden_states[-1][0]  # [0] to get the first item in batch
        else:
            # Fallback to last_hidden_state
            token_embeddings = outputs.last_hidden_state[0]  # [0] to get the first item in batch
            
        # Normalize embeddings for cosine similarity
        token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)
        
        return token_embeddings

    def add_topic(self, topic: Topic) -> None:
        """Register a new topic and pre-compute its token-level embeddings"""
        self.topics[topic.id] = topic
        topic_embeddings = []
        
        # Add topic name and description as first examples
        base_examples = [
            f"{topic.name}: {topic.description}"
        ]
        
        # Combine with provided examples if they exist
        all_examples = base_examples + (topic.examples if topic.examples else [])
        
        # Process each example
        for example in all_examples:
            # Get token-level embeddings
            outputs, offset_mapping = self.embed_full_document(example)
            token_embeddings = self.get_token_embeddings_from_output(outputs)
            
            # Store embeddings and metadata
            topic_embeddings.append({
                'text': example,
                'token_embeddings': token_embeddings,
                'offset_mapping': offset_mapping,
                'is_description': example in base_examples  # Flag to identify description
            })
        
        # Store embeddings for this topic
        self.topic_token_embeddings[topic.id] = topic_embeddings
        self.logger.info(f"Added topic {topic.id}: {topic.name} with {len(topic_embeddings)} embeddings "
                         f"(1 description + {len(topic_embeddings)-1} examples)")

    def compute_topic_score(self, topic_examples, chunk_tokens):
        """
        Compute topic score considering both description and examples.
        Gives higher weight to matches with the topic description.
        """
        description_scores = []
        example_scores = []
        
        # Score against each example for this topic
        for example_data in topic_examples:
            topic_tokens = example_data['token_embeddings']
            if len(topic_tokens) == 0:
                continue
            
            # Compute MaxSim score
            score = self.compute_maxsim_score(topic_tokens, chunk_tokens)
            
            # Store score in appropriate list
            if example_data['is_description']:
                description_scores.append(score)
            else:
                example_scores.append(score)
        
        # Compute final score with weighted combination
        desc_score = max(description_scores) if description_scores else 0
        example_score = max(example_scores) if example_scores else 0
        
        # Weight description matches more heavily (adjustable)
        DESCRIPTION_WEIGHT = 1.5
        EXAMPLE_WEIGHT = 1.0
        
        combined_score = (DESCRIPTION_WEIGHT * desc_score + EXAMPLE_WEIGHT * example_score) / (DESCRIPTION_WEIGHT + EXAMPLE_WEIGHT)
        
        return combined_score

    def get_chunk_token_spans(self, document, offset_mapping, chunks):
        """
        Find token spans for each chunk in the document.
        Uses find_chunk_spans from span_utils for consistent span identification.
        """
        # Get span indices for each chunk
        span_indices = []
        for chunk in chunks:
            # Use find_chunk_spans to locate token positions
            spans = find_chunk_spans(document, offset_mapping, chunk)
            if spans:
                span_indices.append(spans[0])  # Take the first span if multiple found
            else:
                # If no span found, use a dummy span
                print(f"Warning: No span found for chunk '{chunk}'. Using dummy span.")
                span_indices.append((0, 1))
                
        return span_indices
    
    def get_late_chunked_embeddings(self, outputs, sentence, offset_mapping, chunks):
        """
        Extract chunk embeddings using the late chunking approach.
        Integrates properly with the span_utils module.
        """
        # Extract token embeddings
        token_embeddings = self.get_token_embeddings_from_output(outputs)
        
        # Get span indices for each chunk
        span_indices = self.get_chunk_token_spans(sentence, offset_mapping, chunks)
        
        # Extract embeddings for each span using token indices
        chunk_tokens = []
        for start_idx, end_idx in span_indices:
            # Extract the token embeddings for this span
            span_tokens = token_embeddings[start_idx:end_idx]
            chunk_tokens.append(span_tokens)
            
        return chunk_tokens, span_indices

    def compute_maxsim_score(self, query_tokens, doc_tokens):
        """
        Compute MaxSim score between query and document tokens using PyTorch operations.
        Follows the logic of the C++ segmented_maxsim implementation.
        """
        if len(query_tokens) == 0 or len(doc_tokens) == 0:
            return 0.0
            
        # Compute cosine similarity between all token pairs [query_len, doc_len]
        similarity_matrix = torch.matmul(query_tokens, doc_tokens.transpose(0, 1))
        
        # For each query token, find the max similarity across all document tokens
        max_similarities, _ = similarity_matrix.max(dim=1)
        
        # Sum these maxima for the final relevance score (exact match to C++ implementation)
        score = max_similarities.sum().item()

        # Sum the max similarities and normalize by query length
        normalized_score = score / len(query_tokens) if len(query_tokens) > 0 else 0
        
        return normalized_score
    
    def embed_chunks(self, chunks):
        return []

    def predict_clusters_typical(self, chunks, sentence, verbose=False):
        """Match chunks to topics using typical document-level encoding."""
        if not self.topics:
            raise ValueError("No topics registered. Add topics using add_topic() before prediction.")
        
        results = []
        self.logger.info(f"Scoring {len(chunks)} chunks against {len(self.topics)} topics...")
        
        # Process each chunk
        for chunk in tqdm(chunks, desc="Processing chunks", disable=not verbose):
            # Get document embeddings for this chunk
            outputs, _ = self.embed_full_document(chunk)
            chunk_tokens = self.get_token_embeddings_from_output(outputs)
            
            # Score against each topic
            topic_scores = []
            for topic_id, topic_examples in self.topic_token_embeddings.items():
                # Get combined score using description and examples
                score = self.compute_topic_score(topic_examples, chunk_tokens)
                
                if score >= self.relevance_threshold:
                    topic_scores.append((topic_id, score))
            
            # Sort by score
            topic_scores = sorted(topic_scores, key=lambda x: x[1], reverse=True)
            
            # Add results for all matching topics
            for topic_id, score in topic_scores:
                topic = self.topics[topic_id]
                results.append({
                    'text': chunk,
                    'cluster': topic_id,
                    'cluster_title': topic.title,
                    'cluster_summary': topic.summary, 
                    'relevance_score': float(score),
                    'TURN_LEN': len(chunk.split()),
                    'embedding': np.zeros(1)  # Dummy embedding for compatibility
                })
        
        # Return empty DataFrame if no matches
        if not results:
            return pd.DataFrame(columns=['text', 'cluster', 'cluster_title', 'cluster_summary', 'TURN_LEN', 'embedding'])
            
        return pd.DataFrame(results)

    def predict_clusters(self, chunks, sentence, chunks_embs=None, verbose=False):
        """
        Match chunks to topics using either late chunking or typical mode.
        """
        if self.prediction_mode == "typical":
            return self.predict_clusters_typical(chunks, sentence, verbose)
        else:
            # Original late chunking implementation
            return self.predict_clusters_late_chunking(chunks, sentence, verbose)

    def predict_clusters_late_chunking(self, chunks, sentence, verbose=False):
        """Late chunking implementation with improved topic scoring"""
        if not self.topics:
            raise ValueError("No topics registered. Add topics using add_topic() before prediction.")
        
        # Get full document embeddings
        self.logger.info("Embedding full document...")
        outputs, offset_mapping = self.embed_full_document(sentence)
        
        # Extract chunk token embeddings via late chunking
        self.logger.info("Applying late chunking to extract span embeddings...")
        chunk_tokens_list, span_indices = self.get_late_chunked_embeddings(
            outputs, sentence, offset_mapping, chunks
        )
        
        results = []
        self.logger.info(f"Scoring {len(chunks)} chunks against {len(self.topics)} topics...")
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks", disable=not verbose)):
            chunk_tokens = chunk_tokens_list[chunk_idx]
            
            if len(chunk_tokens) == 0:
                continue
                
            # Score against each topic
            topic_scores = []
            for topic_id, topic_examples in self.topic_token_embeddings.items():
                # Get combined score using description and examples
                score = self.compute_topic_score(topic_examples, chunk_tokens)
                
                if score >= self.relevance_threshold:
                    topic_scores.append((topic_id, score))
            
            # Sort by score
            topic_scores = sorted(topic_scores, key=lambda x: x[1], reverse=True)
            
            # Add results for all matching topics
            for topic_id, score in topic_scores:
                topic = self.topics[topic_id]
                results.append({
                    'text': chunk,
                    'cluster': topic_id,
                    'cluster_title': topic.title,
                    'cluster_summary': topic.summary, 
                    'relevance_score': float(score),
                    'TURN_LEN': len(chunk.split()),
                    'embedding': np.zeros(1)
                })
        
        if not results:
            return pd.DataFrame(columns=['text', 'cluster', 'cluster_title', 'cluster_summary', 'TURN_LEN', 'embedding'])
            
        return pd.DataFrame(results)
    
    def predict(
        self, 
        sentence: str, 
        stride: int, 
        min_len: int, 
        max_len: int, 
        sent_id: str, 
        cache_sent: bool = False, 
        verbose: bool = True, 
        top_k: int = 10
    ) -> Tuple[pd.DataFrame, List[int], pd.DataFrame, List[float]]:
        """
        Match text spans to registered topics using late interaction MaxSim scoring.
        Fully integrated with the existing curation framework.
        """
        self.logger.info("---------------------------------------------------")
        self.logger.info(f"Late interaction prediction for sentence ID {sent_id}")
        
        if sent_id not in self.sentences:
            self.config.min_len = min_len
            self.config.max_len = max_len

            self.logger.info(f"Creating overlapping chunks with stride {stride}.")
            # Step 1: Create overlapping chunks with custom stride
            chunks = self.create_overlapping_chunks_with_stride(sentence, stride)

            self.logger.info(f"Filtering chunks by length: {min_len}-{max_len} words.")
            # Step 2: Filter chunks by length
            filtered_chunks = [chunk for chunk in chunks if min_len <= len(chunk.split()) <= max_len]

            self.logger.info(f"Matching {len(filtered_chunks)} chunks to topics using ColBERT-style late interaction.")
            # Match chunks to topics with efficient late interaction
            cluster_predictions = self.predict_clusters(filtered_chunks, sentence, verbose=verbose)

            if cache_sent:
                self.sentences[sent_id] = {
                    'sentence': sentence,
                    'chunks': filtered_chunks,
                    'chunks_embs': None,  # Not storing embeddings to save memory
                    'predicted_df': cluster_predictions
                }
        else:
            # Retrieve cached results
            filtered_chunks = self.sentences[sent_id]['chunks']
            cluster_predictions = self.sentences[sent_id]['predicted_df']
        
        # Handle empty predictions
        if cluster_predictions.empty:
            self.logger.info("No relevant spans found for any topics.")
            return pd.DataFrame(), [], pd.DataFrame(), []
        
        # For compatibility, use Pareto optimization method from base class
        pareto_topic_ids, score_df, scores = self._get_pareto_optimal_clusters(
            cluster_predictions,
            sentence,
            top_k=top_k
        )
        
        if not pareto_topic_ids:
            self.logger.info("No topics passed the pareto optimization threshold.")
            return pd.DataFrame(), [], pd.DataFrame(), []
        
        # Get representative span for the top topic
        self.logger.info(f"Getting best representative span for top topic (ID: {pareto_topic_ids[0]})")
        
        # Get the span with highest score for the top topic
        top_topic_spans = cluster_predictions[cluster_predictions[self.cluster_column] == pareto_topic_ids[0]]
        if not top_topic_spans.empty:
            top_span_idx = top_topic_spans['relevance_score'].idxmax()
            representative_span = top_topic_spans.loc[[top_span_idx]]
        else:
            representative_span = pd.DataFrame()
        
        self.logger.info(f"Found {len(pareto_topic_ids)} relevant topics")
        self.logger.info("---------------------------------------------------")
        
        return representative_span, pareto_topic_ids, score_df, scores
    

if __name__ == "__main__":
    # Usage
    import os
    import torch
    import pandas as pd
    from pathlib import Path
    from datetime import datetime
    from transformers import AutoTokenizer, AutoModel
    from src.embedding.clustering import ClusteringConfig, ClusteringPipeline
    from src.curation.analyzers.local_cluster_analyzer import LocalClusterAnalyzer
    from src.curation.analyzers.base_analyzer import AnalyzerConfig
    from src.curation.analyzers.late_interaction_cross_encoder_analyzer import *
    from src.curation.curator import Curator, CuratorConfig
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)


    topics = [
        Topic(
            id=1, 
            name="Customer Complaints", 
            description="Issues, problems or complaints from customers",
            examples=[
                "I'm having trouble with your product",
                "This service is not working properly",
                "I want to file a complaint about poor service"
            ]
        ),
        Topic(
            id=2, 
            name="Product Inquiries", 
            description="Questions about product features, pricing, or availability",
            examples=[
                "What features does your product have?",
                "Is this item available in blue?",
                "How much does the premium version cost?"
            ]
        ),
        Topic(
            id=3, 
            name="Technical Support", 
            description="Technical issues and troubleshooting help",
            examples=[
                "My application keeps crashing",
                "I'm getting an error message when trying to login",
                "How do I reset my password?"
            ]
        ),
        # unrelated topic
        Topic(
            id=4, 
            name="Geo-Political News",
            description="News and updates on global politics",
            examples=[
                "What is the latest news on the US election?",
                "How are the trade talks between China and the US going?",
                "Is there any update on the Brexit negotiations?"
            ]
        ),
        # unrelated topic
        Topic(
            id=5, 
            name="Sports News",
            description="Latest scores, updates, and events in the sports world",
            examples=[
                "Who won the latest football match?",
                "What are the current standings in the NBA?",
                "Are there any major sports events coming up?"
            ]
        ),
        # markerting topic
        Topic(
            id=6, 
            name="Conversations and analysis around Marketing Campaigns",
            description="Information about marketing campaigns and promotions, how they have performed.",
            examples=[
                "How did the latest marketing campaign perform?",
                "What are the key takeaways from the recent marketing analysis?",
                "What are the best practices for marketing campaigns?",
                "What are the key metrics to track for marketing campaigns?",
                "is there any update on the marketing campaign?"
            ]
        ),
        # forecasting topic
        Topic(
            id=7, 
            name="Enhancing Organizational Forecasting Through Strategic Tool Adoption and Adjustments",
            description="Strategies and tools to improve organizational forecasting and planning.",
            examples=[
                "How can organizations improve their forecasting accuracy?",
                "What are the best tools for organizational forecasting?",
                "How can organizations adjust their forecasting strategies?",
                # much longer example
                "What are the key strategies and tools that organizations can adopt to enhance their forecasting accuracy and planning capabilities?",
                # long, complex, conversational example
                "Can you provide some insights into how organizations can enhance their forecasting accuracy and planning capabilities through the adoption of strategic tools and adjustments to their existing forecasting strategies?"
            ]
        ),
        # financial topic
        Topic(
            id=8, 
            name="Financial Planning and Investment Strategies",
            description="Strategies and tips for financial planning and investment decisions.",
            examples=[
                "What are the best investment strategies for beginners?",
                "How can I improve my financial planning skills?",
                "What are the key factors to consider when making investment decisions?"
            ]
        ),
    ]

    # Load Jina model
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

    # Initialize analyzer with Jina model
    analyzer = LateInteractionCrossEncoder(
        config=AnalyzerConfig(min_len=5, max_len=50),
        topics=topics,
        cache_save_dir="cache/jina_late_interaction",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={
            'model_batch_size': 32,
            'lora_task': 'retrieval.passage'  # Set to appropriate task if using adapters
        },
        relevance_threshold=0.25,
        prediction_mode='late_chunking'
    )

    # 6. Test Data Preparation
    logger.info("Loading test corpus...")
    try:
        test_corpus = pd.read_csv("src/curation/analyzers/sample_texts.csv")
        text = "".join(test_corpus['text'].values[4:5])
        label = "|".join(test_corpus['topic'].values[4:5])
        
        logger.debug(f"Selected text length: {len(text.split())} words")
        logger.debug(f"Text label: {label}")
    except Exception as e:
        logger.error(f"Failed to load test corpus: {str(e)}")
        raise


    text = """Yeah, generally. I'm trying to remember what we would do at Salesforce. I think we had, I don't know, six or so stages. And then stage zero, one was just an idea in your head. You haven't even vocalized it to the customer. So it's definitely not a thing. And then stage two is maybe like, okay, we talked with the customer, that could be pipeline if you're feeling optimistic about it. And then from there, as we progressed, it would go more best case, commit, definitely by the time a quote was out or if a demo had been done. And again, feeling confident, good feedback from customer. I don't know, Patrick, what have you seen out in the field?. Yeah, I mean, pipeline, at least to delineate, pipeline as in the forecast category here, this is where it's very, very early stage deals. But then I think depending on where you are in the quarter, that's where I would say that, hey, these deals should... You should no longer have pipeline deals in your book of business. If you have two weeks left in the quarter and you have a bunch of deals where forecast category equals pipeline, then with just two weeks left in the quarter of this quarter close date, then it doesn't make sense. Either it's a best case or commit deal, it should no longer be a pipeline deal. So now you got to push. Now that's when you start doing your pipeline, your opportunity, clean up your hygiene to move that to a next quarter and just say, hey, is this actually a best case deal for the next quarter or is it a commit deal for the next quarter? Just kind of in our experiences that depending at a certain time of the quarter, there should no longer be pipeline deals in your book of business because it's either going to close that quarter or it's not. So if it's not going to close that quarter, either you lost it or if you're still working it, then you need to move it to a quarter where it's realistically sorry, realistically going, you think it's going to close."""

    # Run prediction
    representative_span, matched_topic_ids, score_df, scores = analyzer.predict(
        sentence=text,
        stride=2,
        min_len=20, 
        max_len=100,
        sent_id="sample_turn_1",
        cache_sent=True,
        top_k=5
    )

    # Print results
    print(f"Best matching span: {representative_span[analyzer.text_column].values[0]}")
    print("\nMatched topics:")
    for i, topic_id in enumerate(matched_topic_ids):
        topic = analyzer.topics[topic_id]
        print(f"{i+1}. {topic.name} (Score: {scores[i]:.2f})")


    analyzer.analyze("sample_turn_1", matched_topic_ids)
