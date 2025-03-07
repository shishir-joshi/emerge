from typing import Generator, List
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from IPython.display import display, HTML

def streaming_embeddings(
    texts: List[str], model: AutoModel = None, tokenizer: AutoTokenizer = None, device: str = "cuda", task: str = 'separation', batch_size: int = 32, display_mem_usage: bool = False
) -> Generator[np.ndarray, None, None]:
    """
    Generate embeddings for a list of texts in a streaming fashion (one batch at a time, offloading to CPU if necessary).
    Useful for large datasets that do not fit in GPU memory.

    Currently supports LoRA embeddings for adapter-based models (ref: jina-embedding-v3).
    """
    # LoRA setup
    adapter_mask = None
    if task:
        task_id = model._adaptation_map[task]
        num_examples = 1 if isinstance(texts, str) else len(texts)
        adapter_mask = torch.full(
            (num_examples,), task_id, dtype=torch.int32, device=device
        )
        if isinstance(texts, str):
            texts = model._task_instructions[task] + texts
        else:
            texts = [
                model._task_instructions[task] + tx for tx in texts
            ]
    
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Tokenize and move to GPU
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        ).to(device)
    
        lora_arguments = (
            {"adapter_mask": adapter_mask[i : i + batch_size]}
            if adapter_mask is not None
            else {}
        )

        # model(**inputs)
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model.forward(**inputs, **lora_arguments)[0]
        
        # Saving on precision
        # outputs = outputs.float()
        
        # mean pooling
        outputs = model.roberta.mean_pooling(
            outputs, inputs["attention_mask"]
        )
        
        # normalization
        outputs = torch.stack(
            [torch.nn.functional.normalize(embedding, p=2, dim=0) 
             for embedding in outputs
            ]
        )
        
        # Monitor GPU memory usage
        if display_mem_usage:
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        if device != "cpu":
            # Move to CPU and convert to numpy
            cpu_embeddings = outputs.detach().cpu().numpy()
            yield cpu_embeddings
        else:
            yield outputs.numpy()

        # Memory cleanup
        del inputs, outputs
        torch.cuda.empty_cache()

        
# Helper function to get embeddings for a list of texts, return as torch tensor
def get_embeddings(texts: List[str], model: AutoModel = None, tokenizer: AutoTokenizer = None, device: str = "cuda", task: str = 'separation', batch_size: int = 32) -> torch.Tensor:
    embeddings = []
    for batch_embeddings in streaming_embeddings(texts, model, tokenizer, device, task, batch_size):
        embeddings.extend(batch_embeddings)
    return torch.tensor(embeddings)


if __name__ == "__main__":
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    texts = [
        "Hello, my name is John.",
        "I like to play soccer.",
        "This is a test sentence."
    ]

    for embeddings in streaming_embeddings(texts, model, tokenizer):
        print(embeddings.shape)
        print(embeddings)
        print()