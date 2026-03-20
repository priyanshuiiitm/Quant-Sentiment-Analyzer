import torch
import numpy as np
from transformers import AutoTokenizer,AutoModel
tokenizer=AutoTokenizer.from_pretrained("ProsusAI/finbert",local_files_only=True)
model=AutoModel.from_pretrained("ProsusAI/finbert",local_files_only=True)
model.eval()
def encode_headlines(headlines):
    clean_headlines=[h.strip() for h in headlines if isinstance(h,str) and h.strip()]
    if not clean_headlines:
        return np.zeros(768,dtype=np.float32)
    inputs=tokenizer(
        clean_headlines,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs=model(**inputs)
    embeddings=outputs.last_hidden_state[:,0,:].numpy()
    daily_embedding=embeddings.mean(axis=0).astype(np.float32)
    return daily_embedding
