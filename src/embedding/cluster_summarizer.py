import json
import pandas as pd
import tiktoken
import traceback
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
from src.utils.openai_api_parallelizer import chat
from src.utils.async_openai_api_utils import *

@dataclass
class ModelConfig:
    """Configuration for OpenAI model parameters"""
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    response_format: Dict[str, str] = None
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'stop': self.stop,
            'response_format': {"type": "json_object"},
            'seed': self.seed
        }

class ClusterSummarizer:
    """Main class for handling cluster summarization"""
    
    def __init__(self, model: str = "gpt-4", batch_size: int = 20):
        self.model = model
        self.batch_size = batch_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self._initialize_model_configs()
        
    def _initialize_model_configs(self):
        """Initialize configurations for different models"""
        self.model_configs = {
            "gpt-3.5-turbo": ModelConfig(temperature=0, max_tokens=2000, top_p=1),
            "gpt-3.5-turbo-16k": ModelConfig(temperature=0, max_tokens=2000, top_p=1),
            "gpt-4": ModelConfig(temperature=1, max_tokens=2000, top_p=0.73),
            "gpt-4-turbo": ModelConfig(temperature=1, max_tokens=3000, top_p=1),
            "gpt-4-turbo-preview": ModelConfig(temperature=1, max_tokens=3000, top_p=1),
            "gpt-4-1106-preview": ModelConfig(temperature=1, max_tokens=3000, top_p=1),
            "gpt-4-32k": ModelConfig(temperature=0, max_tokens=3000, top_p=1),
            "gpt-4-turbo-2024-04-09": ModelConfig(temperature=1, max_tokens=3000, top_p=1),
            "gpt-4o-2024-05-13": ModelConfig(temperature=1, max_tokens=3000, top_p=1)
        }

    @property
    def system_message(self) -> str:
        """Get the system message for cluster summarization"""
        return """
        # Cluster Summary and Analysis Task
You are given a series of sentences, each labeled with a unique turn ID. Your task is to analyze these sentences and generate a summary of the main topics or key phrases mentioned. Follow these steps:

## Step 1: Initial Analysis
- Read through all sentences carefully.
- Identify main topics or key phrases in each sentence.
- Exclude personal names and pleasantries; focus only on significant topics.
- Output: List 3-5 examples of main topics or key phrases identified.

## Step 2: Concise Summary
- Write a concise summary of the identified topics.
- The summary should not exceed 50 words.
- Output: The summary.

## Step 3: Topic Clustering
- Group sentences into clusters based on cohesive topics.
- Avoid creating clusters around topics relevant to only one or two sentences.
- Incorporate minor topics into broader clusters.
- Output: List 2-3 examples of topics and the main topic or key phrase they map to.

## Step 4: Source Citation
- Cite sources using the turn ID for each sentence.
- Use the format [number] for citation.
- For consecutive turns, use a number range like [0-5, 23-27].
- Output: 2-3 examples of how you've cited sources.

## Step 5: Outlier Identification
- Identify sentences that don't fit well into any identified clusters.
- These sentences may deviate from the main topics or themes.
- Output: List of turn IDs for potential outliers, if any. If no outliers, state "No outliers identified."

## Step 6: Cluster Title Generation
- Create a specific and descriptive title for the cluster.
- Use concrete nouns and active verbs related to the cluster's content.
- Aim for a title that could be used as a headline or document title.
- The title should be extremely specific and descriptive to this cluster.
- Output: The cluster title.

(Note: Put the output for each step inside the "reasoning_steps" field as shown in the output format below. Do not output anything other than the specified JSON format. Ensure all field names in the JSON output exactly match those specified below.)

Output format (JSON only):
{
 "reasoning_steps": [
 "Step 1 output: <output of this step>",
 "Step 2 output: <output of this step>",
 "Step 3 output: <output of this step>",
 "Step 4 output: <output of this step>",
 "Step 5 output: <output of this step>",
 "Step 6 output: <output of this step>"
 ],
 "topics": [
 "topic 1 [turn IDs]",
 "topic 2 [turn IDs]",
 "topic 3 [turn IDs]",
 ...
 ],
 "summary": "<summary in no more than 50 words>",
 "cluster_title": "<a title for the cluster which is derived from the topics identified in this cluster>",
 "potential_outliers": "<list of turn IDs that don't fit well into any topic, or 'No outliers identified'>"
}
"""

    def summarize_clusters(self, 
                         df: pd.DataFrame, 
                         text_column: str, 
                         cluster_column: str,
                         max_samples: int = 100) -> Tuple[Dict, Dict, Dict]:
        """Main method to summarize all clusters"""
        try:
            cluster_summaries = self._get_all_summaries_parallel(
                df, text_column, cluster_column, max_samples
            )
            parsed_summaries = self._parse_summaries(cluster_summaries)
            formatted_summaries = self._format_summaries(parsed_summaries)
            return cluster_summaries, parsed_summaries, formatted_summaries
        except Exception as e:
            print(f"Error in summarization pipeline: {e}")
            traceback.print_exc()
            return None, None, None

    def _get_all_summaries_parallel(self, 
                                  df: pd.DataFrame, 
                                  text_col: str, 
                                  cluster_col: str,
                                  max_samples: int) -> Dict:
        """Get summaries for all clusters in parallel"""
        try:
            cluster_ids = df[cluster_col].unique()
            prompts = []
            id_map = []

            for cluster_id in cluster_ids:
                context = self._prepare_cluster_context(
                    df, cluster_id, text_col, cluster_col, max_samples
                )
                prompts.append(context)
                id_map.append(cluster_id)

            # Create chat config dictionary in the expected format
            chat_config = {
                self.model: self.model_configs[self.model].to_dict()
            }
            
            results = chat(
                prompts=prompts,
                system_prompt=self.system_message,
                chat_model=self.model,
                chat_config=chat_config,  # Pass wrapped config
                batch_size=self.batch_size
            )

            return {
                cluster_id: response 
                for (_, [response], _), cluster_id in zip(results, id_map)
            }
        except Exception as e:
            print(f"Error in parallel summarization: {str(e)}")
            traceback.print_exc()
            return {}

    def _prepare_cluster_context(self, 
                               df: pd.DataFrame, 
                               cluster_id: Any,
                               text_col: str, 
                               cluster_col: str, 
                               max_samples: int) -> str:
        """Prepare context for a single cluster"""
        cluster_df = df[df[cluster_col] == cluster_id].reset_index(drop=True).head(
            n=max_samples if max_samples <= df[df[cluster_col] == cluster_id].shape[0]
            else df[df[cluster_col] == cluster_id].shape[0]
        )
        texts = cluster_df[text_col].values.tolist()
        return "\n".join(
            f"### TURN ID {i} {text}" for i, text in enumerate(texts)
        )

    def _fix_summary_formatting(self, broken_summ: str) -> str:
        """Fix broken JSON formatting in summaries"""
        resp = chat(
            system_prompt="Please fix this JSON formatting, output only the corrected json:",
            prompts=[broken_summ],
            chat_model=self.model,
            chat_config=self.model_configs[self.model].to_dict(),
        )
        return resp[0][1][0]

    def _parse_summaries(self, cluster_summaries: Dict) -> Dict:
        """Parse raw summaries into structured format"""
        parsed_summaries = {}
        for k, v in cluster_summaries.items():
            v = v.replace('```json', '').replace('```', '')
            try:
                parsed_summaries[int(k)] = eval(v)
            except Exception as e:
                print(f"Failed to parse JSON for cluster {k}: {e}")
                try:
                    fixed_v = self._fix_summary_formatting(v)
                    parsed_summaries[int(k)] = json.loads(fixed_v)
                except Exception as e:
                    print(f"Failed to fix JSON for cluster {k}: {e}")
                    parsed_summaries[int(k)] = None
        return parsed_summaries

    def _format_summaries(self, parsed_summaries: Dict) -> Dict:
        """Format parsed summaries for display"""
        formatted = {}
        for cluster_id, summary in parsed_summaries.items():
            if not summary:
                continue
            try:
                    #                 Reasoning Steps: {f'{chr(10)}{chr(10)}:'.join(summary['reasoning_steps'])}
                    # ---------------------------------
                formatted[cluster_id] = f"""
Title: {summary['cluster_title']} 
---------------------------------
Description: {summary['summary']} 
---------------------------------
Topics: 
:{f'\t{chr(10)}{chr(10)}:'.join(summary['topics'])} 
----------------
                """
            except Exception as e:
                print(f"Error formatting cluster {cluster_id}: {e}")
        return formatted

    def print_summaries(self, formatted_summaries: Dict):
        """Print formatted summaries in a readable way"""
        print("##", "\n\n## ".join([
            f"id: {cid}, {c}" 
            for cid, c in sorted(formatted_summaries.items(), key=lambda x: int(x[0]))
        ]))

# For backward compatibility
def get_all_cluster_summaries_parallel(*args, **kwargs):
    summarizer = ClusterSummarizer()
    return summarizer.summarize_clusters(*args, **kwargs)


if __name__ == "__main__":
    try:
        # Initialize summarizer with specific model config
        summarizer = ClusterSummarizer(model="gpt-4o-2024-05-13", batch_size=20)

        # Create sample data with more complex topics and multiple clusters
        df = pd.DataFrame({
            'text': [
            "The new quantum computing breakthrough enables faster drug discovery processes",
            "CRISPR gene editing shows promise in treating genetic disorders",
            "Machine learning algorithms improve cancer detection accuracy by 45%",
            "Researchers develop self-healing materials for sustainable construction",
            "Artificial neural networks revolutionize weather prediction models",
            "Blockchain technology enhances supply chain transparency in pharmaceutical industry",
            "New nanotechnology applications in targeted drug delivery systems",
            "Quantum entanglement demonstrates potential in secure communications",
            "Smart materials respond to environmental changes for better energy efficiency",
            "AI-powered diagnostic tools reduce medical imaging analysis time by 60%",
            "Robotic surgery systems achieve unprecedented precision in microsurgery",
            "Advanced battery technology extends electric vehicle range to 600 miles",
            "3D bioprinting creates functional human tissue for transplantation",
            "Sustainable hydrogen production breakthrough using solar catalysts",
            "Neural interfaces allow direct brain-computer communication"
            ],
            'cluster': [0, 1, 2, 3, 2, 4, 1, 0, 3, 2, 1, 4, 1, 4, 0]
        })

        # Get summaries with error handling
        summaries, parsed, formatted = summarizer.summarize_clusters(
            df,
            text_column='text',
            cluster_column='cluster',
            max_samples=100
        )

        if formatted:
            summarizer.print_summaries(formatted)
        else:
            print("No valid summaries generated")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()