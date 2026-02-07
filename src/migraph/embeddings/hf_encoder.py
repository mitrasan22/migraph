from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from migraph.embeddings.encoder import EmbeddingEncoder


class HuggingFaceEmbeddingEncoder(EmbeddingEncoder):
    """
    HuggingFace-based sentence embedding encoder.

    This encoder:
    - uses mean pooling over token embeddings
    - is deterministic
    - is backend-configured
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize = normalize

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        model.to(device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

        # Infer embedding dimension dynamically
        with torch.no_grad():
            dummy = tokenizer("test", return_tensors="pt").to(device)
            out = model(**dummy)
            dim = out.last_hidden_state.shape[-1]

        super().__init__(dimension=dim)

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _encode_one(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a sentence embedding.
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)

            # Mean pooling (mask-aware)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)

            embedding = summed / counts

        vec = embedding.squeeze(0).cpu().numpy()

        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        return vec
