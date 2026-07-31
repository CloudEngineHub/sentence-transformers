from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers.base.losses.merged_forward import embed_columns_padded
from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder
from sentence_transformers.util.similarity import maxsim_pairwise


class MultiVectorMarginMSELoss(nn.Module):
    """Margin-MSE distillation loss for :class:`~sentence_transformers.MultiVectorEncoder` models.

    Adapted from the dense :class:`sentence_transformers.losses.MarginMSELoss`. Given a query, a positive
    document, and one or more negative documents, plus teacher margins ``score(q, pos) - score(q, neg)``,
    the student's MaxSim margins are MSE-matched to the teacher's.

    Two label formats are supported:

    1. **Single negative**: ``sentence_features = (query, positive, negative)`` with ``labels`` of shape
       ``(batch_size,)`` containing the teacher margin ``s(q, pos) - s(q, neg)``.
    2. **Multiple negatives**: ``sentence_features = (query, positive, negative_1, ..., negative_k)`` with
       ``labels`` of shape ``(batch_size, k)`` containing per-negative teacher margins.

    Args:
        model: A :class:`~sentence_transformers.MultiVectorEncoder`.
        similarity_fct: A pairwise scoring function. Defaults to
            :func:`~sentence_transformers.util.maxsim_pairwise`.
        size_average: ``True`` (default) averages the MSE across the batch. ``False`` sums.
        mini_batch_size: Maximum number of rows per model forward. The merged document batch is
            split into row chunks, each re-trimmed to its own longest document, so a single long
            outlier document only widens its own chunk. Chunking is exact for this loss (no
            cross-row interactions). ``None`` (default) runs one merged forward.

    Example:
        ::

            from datasets import Dataset

            from sentence_transformers import MultiVectorEncoder, MultiVectorEncoderTrainer
            from sentence_transformers.multi_vector_encoder.losses import MultiVectorMarginMSELoss

            model = MultiVectorEncoder("answerdotai/ModernBERT-base")
            # label is the teacher margin score(query, positive) - score(query, negative), so a
            # positive value means the teacher ranked the positive above the negative.
            train_dataset = Dataset.from_dict(
                {
                    "query": ["What is the capital of France?", "Who painted the Mona Lisa?"],
                    "positive": ["Paris is the capital of France.", "Leonardo da Vinci painted the Mona Lisa."],
                    "negative": ["Berlin is the capital of Germany.", "Van Gogh painted The Starry Night."],
                    "label": [3.5, 2.8],
                }
            )
            loss = MultiVectorMarginMSELoss(model)

            trainer = MultiVectorEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
            trainer.train()
    """

    # Enables per-sample media counting in Transformer.preprocess, so mini-batching can slice VLM
    # inputs (e.g. Qwen2-VL's flattened pixel_values) along the batch dimension.
    requires_media_counts = True

    def __init__(
        self,
        model: MultiVectorEncoder,
        similarity_fct: Callable | None = None,
        size_average: bool = True,
        mini_batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct if similarity_fct is not None else maxsim_pairwise
        self.size_average = size_average
        self.mini_batch_size = mini_batch_size
        self.loss_function = nn.MSELoss(reduction="mean" if size_average else "sum")

    def get_config_dict(self) -> dict[str, Any]:
        similarity_fct = getattr(self.similarity_fct, "__name__", type(self.similarity_fct).__name__)
        # Configured metric objects expose their own config, include it.
        metric_config = getattr(self.similarity_fct, "get_config_dict", None)
        if metric_config is not None:
            args = ", ".join(f"{key}={value!r}" for key, value in metric_config().items())
            similarity_fct = f"{similarity_fct}({args})"
        return {
            "similarity_fct": similarity_fct,
            "size_average": self.size_average,
            "mini_batch_size": self.mini_batch_size,
        }

    def _score(
        self, query_embeddings: Tensor, document_embeddings: Tensor, query_mask: Tensor, document_mask: Tensor
    ) -> Tensor:
        return self.similarity_fct(query_embeddings, document_embeddings, a_mask=query_mask, b_mask=document_mask)

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor,
    ) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) < 3:
            raise ValueError(
                f"{type(self).__name__} expects at least 3 sentence features (query, positive, negative), but "
                f"got {len(sentence_features)}."
            )

        # Collator-stamped tasks (positional fallback), masks from the model output where
        # MultiVectorMask has rewritten attention_mask into the per-row scoring mask.
        query_features = sentence_features[0]
        query_outputs = self.model(query_features, task=query_features.get("task", "query"))
        q = query_outputs["token_embeddings"]
        q_mask = query_outputs["attention_mask"].bool()

        embeddings, masks = embed_columns_padded(
            self.model, sentence_features[1:], self.mini_batch_size, task_default="document"
        )

        pos, *negs = embeddings
        pos_mask, *neg_masks = masks
        pos_scores = self._score(q, pos, q_mask, pos_mask)

        if labels.ndim == 1:
            if len(negs) != 1:
                raise ValueError(
                    f"{type(self).__name__} got 1D labels (shape {tuple(labels.shape)}) but "
                    f"{len(negs)} negative columns (expected 1)."
                )
            neg_scores = self._score(q, negs[0], q_mask, neg_masks[0])
            student_margin = pos_scores - neg_scores
            return self.loss_function(student_margin, labels.to(student_margin.dtype))

        if labels.ndim != 2 or labels.shape[1] != len(negs):
            raise ValueError(
                f"{type(self).__name__} got labels with shape {tuple(labels.shape)} but {len(negs)} "
                f"negative columns. Expected labels of shape (batch_size, {len(negs)})."
            )
        student_margins = torch.stack(
            [pos_scores - self._score(q, n, q_mask, nm) for n, nm in zip(negs, neg_masks)], dim=1
        )
        return self.loss_function(student_margins, labels.to(student_margins.dtype))

    @property
    def citation(self) -> str:
        return """
@misc{hofstaetter2020improving,
    title={Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation},
    author={Sebastian Hofstätter and Sophia Althammer and Michael Schröder and Mete Sertkan and Allan Hanbury},
    year={2020},
    eprint={2010.02666},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
"""
