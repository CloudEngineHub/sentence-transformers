from __future__ import annotations

import pytest
import torch

from sentence_transformers.sentence_transformer.losses import DistillKLDivLoss


def test_distill_kl_div_separate_temperatures() -> None:
    """student_temperature / teacher_temperature default to the shared temperature and split the two
    softmaxes when set, with the loss scaled by the student temperature squared."""
    generator = torch.Generator().manual_seed(11)
    embeddings = [torch.randn(4, 8, generator=generator) for _ in range(3)]
    labels = torch.tensor([[4.0, 1.0], [3.5, 0.5], [2.0, 1.5], [5.0, 0.0]])

    shared = DistillKLDivLoss(model=None, temperature=0.5)
    shared_value = shared.compute_loss_from_embeddings(embeddings, labels).item()
    aliased = DistillKLDivLoss(model=None, student_temperature=0.5, teacher_temperature=0.5)
    assert aliased.compute_loss_from_embeddings(embeddings, labels).item() == pytest.approx(shared_value)

    split = DistillKLDivLoss(model=None, student_temperature=0.5, teacher_temperature=2.0)
    assert split.compute_loss_from_embeddings(embeddings, labels).item() != pytest.approx(shared_value)
    assert split.get_config_dict() == {
        "similarity_fct": "pairwise_dot_score",
        "temperature": 1.0,
        "student_temperature": 0.5,
        "teacher_temperature": 2.0,
    }
