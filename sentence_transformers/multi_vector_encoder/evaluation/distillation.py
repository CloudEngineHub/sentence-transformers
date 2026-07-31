from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING, Any

import torch
from scipy.stats import spearmanr

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator

if TYPE_CHECKING:
    from sentence_transformers.base.modality_types import SingleInput
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder

logger = logging.getLogger(__name__)


class MultiVectorDistillationEvaluator(BaseEvaluator):
    """Distillation evaluator for :class:`~sentence_transformers.MultiVectorEncoder` models.

    Two data shapes are supported:

    - **Per-query candidate sets** (the KD training format, matching
      :class:`~sentence_transformers.multi_vector_encoder.losses.MultiVectorDistillKLDivLoss` and
      PyLate): ``documents`` is a list of N-way candidate lists per query and ``scores`` the matching
      2-D teacher scores. Both metrics are computed per query, so they track the training loss: the KL
      divergence over each query's own candidate set (with the same optional min-max normalization and
      ``temperature`` handling as the loss, averaged over queries, so with the loss's temperature the
      reported KL matches the training loss at its default ``size_average=True``) and the Spearman as
      the mean of the per-query rank correlations.
    - **Flat pairs**: one document per query with 1-D scores. A per-query distribution or ranking is
      undefined here, so the KL softmaxes over the whole dataset as a single distribution and reports
      the full divergence (a sum over pairs, not divided by their number, and not comparable to the
      per-query KL or to PyLate). ``temperature`` divides the scores the same way as on the per-query
      path. The Spearman is one global correlation over all pairs.

    Reported metrics:

    - KL divergence between teacher and student score distributions (lower is better).
    - Spearman rank correlation between teacher and student scores (higher is better, the primary
      metric). With candidate sets this is the mean of the per-query correlations: MaxSim scores are
      summed over query tokens, so absolute scores are not comparable across queries and a single
      correlation over all scores would mostly measure per-query offsets rather than ranking. Queries
      where the teacher or student scores are constant have no defined rank correlation and are
      skipped from the mean (0.0 is reported if every query is skipped).

    The Spearman score is generally a more interpretable mid-training signal than raw KL.

    Args:
        queries: List of query texts.
        documents: One document per query, or a list of N candidate documents per query (N constant
            across queries). Must have the same length as ``queries``.
        scores: Teacher scores: 1-D (one per pair) for flat documents, 2-D ``(num_queries, N)`` for
            candidate sets.
        normalize_scores: Min-max normalize the student scores per query before the softmax, matching
            ``MultiVectorDistillKLDivLoss``. Only used for per-query candidate sets. Defaults to True.
        temperature: Temperature dividing the teacher scores and (normalized) student scores before
            the softmax, with the KL scaled by ``temperature ** 2``, exactly as in
            ``MultiVectorDistillKLDivLoss``. Set it to the training loss's temperature so the
            per-query KL matches the training loss at its default ``size_average=True``. Must be > 0.
            Defaults to 1.0.
        name: Optional run name appended to CSV filenames.
        batch_size: Batch size for encoding.
        show_progress_bar: Whether to show a progress bar.
        write_csv: Whether to write per-call results to a CSV file under ``output_path``.

    Example:
        ::

            from sentence_transformers import MultiVectorEncoder
            from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorDistillationEvaluator

            model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")

            queries = ["What is the capital of France?", "Who painted the Mona Lisa?"]
            # One candidate list per query, with the matching teacher scores.
            documents = [
                ["Paris is the capital of France.", "Berlin is the capital of Germany."],
                ["Leonardo da Vinci painted the Mona Lisa.", "Van Gogh painted The Starry Night."],
            ]
            scores = [[9.5, 2.1], [8.8, 1.4]]

            evaluator = MultiVectorDistillationEvaluator(
                queries=queries,
                documents=documents,
                scores=scores,
                # Match the training loss's temperature so the reported KL tracks the training loss.
                temperature=0.25,
                name="msmarco-dev",
            )
            results = evaluator(model)
            print(results[evaluator.primary_metric])
    """

    def __init__(
        self,
        queries: list[SingleInput],
        documents: list[SingleInput] | list[list[SingleInput]],
        scores: list[float] | list[list[float]] | torch.Tensor,
        normalize_scores: bool = True,
        temperature: float = 1.0,
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ) -> None:
        super().__init__()
        if not (len(queries) == len(documents) == len(scores)):
            raise ValueError(
                f"queries ({len(queries)}), documents ({len(documents)}), and scores ({len(scores)}) "
                f"must all have the same length."
            )
        self.queries = list(queries)
        self.nested_documents = bool(documents) and isinstance(documents[0], (list, tuple))
        if self.nested_documents:
            n_ways = len(documents[0])
            if any(len(row) != n_ways for row in documents):
                raise ValueError("All per-query candidate lists in documents must have the same length.")
            self.documents = [list(row) for row in documents]
            self.scores = torch.as_tensor(scores, dtype=torch.float32)
            if self.scores.ndim != 2 or self.scores.shape[1] != n_ways:
                raise ValueError(
                    f"With per-query candidate documents, scores must be 2-D (num_queries, {n_ways}), "
                    f"got shape {tuple(self.scores.shape)}."
                )
        else:
            self.documents = list(documents)
            self.scores = torch.as_tensor(scores, dtype=torch.float32)
            if self.scores.ndim != 1:
                raise ValueError(
                    f"With one document per query, scores must be 1-D, got shape {tuple(self.scores.shape)}. "
                    "Pass documents as a list of per-query candidate lists to use 2-D scores."
                )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        self.normalize_scores = normalize_scores
        self.temperature = temperature
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv

        self.csv_file = "multi_vector_distillation_evaluation" + (f"_{name}" if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "kl_divergence", "spearman"]

    def __call__(
        self,
        model: MultiVectorEncoder,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        if epoch != -1:
            out_txt = f" after epoch {epoch}" if steps == -1 else f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        logger.info(f"MultiVectorDistillationEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        query_embeddings = model.encode_query(
            self.queries,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )
        if self.nested_documents:
            n_ways = len(self.documents[0])
            flat_documents = [document for row in self.documents for document in row]
            flat_doc_embeddings = model.encode_document(
                flat_documents,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
            )
            # Regroup per way and score pairwise with the model's own similarity, giving the same
            # (num_queries, n_ways) student scores as the KD loss computes at training time.
            pairwise = model.similarity_pairwise
            student_scores = torch.stack(
                [pairwise(query_embeddings, flat_doc_embeddings[way::n_ways]).cpu() for way in range(n_ways)],
                dim=1,
            )
            student_logits = student_scores
            if self.normalize_scores:
                # Same per-query min-max as MultiVectorDistillKLDivLoss.
                max_scores = student_logits.max(dim=1, keepdim=True).values
                min_scores = student_logits.min(dim=1, keepdim=True).values
                student_logits = (student_logits - min_scores) / (max_scores - min_scores + 1e-8)
            teacher_log_probs = torch.log_softmax(self.scores / self.temperature, dim=-1)
            student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        else:
            doc_embeddings = model.encode_document(
                self.documents,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
            )
            student_scores = model.similarity_pairwise(query_embeddings, doc_embeddings).cpu()
            # One document per query: a per-query distribution is undefined, so this softmaxes over
            # the whole dataset (a single global distribution). Not comparable to the per-query KL.
            teacher_log_probs = torch.log_softmax(self.scores / self.temperature, dim=-1)
            student_log_probs = torch.log_softmax(student_scores / self.temperature, dim=-1)

        # batchmean averages the per-query KL over queries. On the flat path it would divide the
        # single global KL by the number of pairs, so report the full sum there instead.
        reduction = "batchmean" if self.nested_documents else "sum"
        kl = torch.nn.functional.kl_div(student_log_probs, teacher_log_probs, reduction=reduction, log_target=True)
        kl = kl.item() * self.temperature**2
        if self.nested_documents:
            # Rank within each query, matching the per-query KL above: absolute MaxSim scores are
            # not comparable across queries (see the class docstring).
            per_query = []
            for teacher_row, student_row in zip(self.scores.numpy(), student_scores.numpy()):
                correlation, _ = spearmanr(teacher_row, student_row)
                if correlation == correlation:  # NaN if either row is constant, skip those queries
                    per_query.append(correlation)
            spearman = sum(per_query) / len(per_query) if per_query else 0.0
        else:
            spearman, _ = spearmanr(self.scores.numpy(), student_scores.numpy())
            if spearman != spearman:  # NaN if all scores are constant
                spearman = 0.0

        metrics = {"kl_divergence": kl, "spearman": float(spearman)}
        logger.info(f"KL divergence:\t{kl:.4f}")
        logger.info(f"Spearman correlation:\t{spearman:.4f}")

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self.csv_headers)
                writer.writerow([epoch, steps, kl, spearman])

        self.primary_metric = "spearman"
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = {}
        if self.temperature != 1.0:
            config_dict["temperature"] = self.temperature
        # normalize_scores does nothing on the flat path.
        if self.nested_documents and not self.normalize_scores:
            config_dict["normalize_scores"] = self.normalize_scores
        return config_dict
