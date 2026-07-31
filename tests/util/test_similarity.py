from __future__ import annotations

import numpy as np
import pytest
import sklearn
import torch

from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.util.similarity import (
    _document_chunk_ranges,
    cos_sim,
    dot_score,
    euclidean_sim,
    manhattan_sim,
    maxsim,
    maxsim_pairwise,
    pairwise_angle_sim,
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
    pytorch_cos_sim,
)


def test_pytorch_cos_sim() -> None:
    """Tests the correct computation of pytorch_cos_sim"""
    a = np.random.randn(50, 100)
    b = np.random.randn(50, 100)

    sklearn_pairwise = sklearn.metrics.pairwise.cosine_similarity(a, b)
    pytorch_cos_scores = pytorch_cos_sim(a, b).numpy()
    for i in range(len(sklearn_pairwise)):
        for j in range(len(sklearn_pairwise[i])):
            assert abs(sklearn_pairwise[i][j] - pytorch_cos_scores[i][j]) < 0.001


def test_pairwise_cos_sim() -> None:
    a = np.random.randn(50, 100)
    b = np.random.randn(50, 100)

    # Pairwise cos
    sklearn_pairwise = 1 - sklearn.metrics.pairwise.paired_cosine_distances(a, b)
    pytorch_cos_scores = pairwise_cos_sim(a, b).numpy()

    assert np.allclose(sklearn_pairwise, pytorch_cos_scores)


def test_pairwise_euclidean_sim() -> None:
    a = np.array([[1, 0], [1, 1]], dtype=np.float32)
    b = np.array([[0, 0], [0, 0]], dtype=np.float32)

    euclidean_expected = np.array([-1.0, -np.sqrt(2.0)])
    euclidean_calculated = pairwise_euclidean_sim(a, b).numpy()

    assert np.allclose(euclidean_expected, euclidean_calculated)


def test_pairwise_manhattan_sim() -> None:
    a = np.array([[1, 0], [1, 1]], dtype=np.float32)
    b = np.array([[0, 0], [0, 0]], dtype=np.float32)

    manhattan_expected = np.array([-1.0, -2.0])
    manhattan_calculated = pairwise_manhattan_sim(a, b).numpy()

    assert np.allclose(manhattan_expected, manhattan_calculated)


def test_pairwise_dot_score_cos_sim() -> None:
    a = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
    b = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)

    dot_and_cosine_expected = np.array([1.0, 0.0, -1.0])
    cosine_calculated = pairwise_cos_sim(a, b)
    dot_calculated = pairwise_dot_score(a, b)

    assert np.allclose(cosine_calculated, dot_and_cosine_expected)
    assert np.allclose(dot_calculated, dot_and_cosine_expected)


def test_euclidean_sim() -> None:
    a = np.array([[1, 0], [0, 1]], dtype=np.float32)
    b = np.array([[0, 0], [0, 1]], dtype=np.float32)

    euclidean_expected = np.array([[-1.0, -np.sqrt(2.0)], [-1.0, 0.0]])
    euclidean_calculated = euclidean_sim(a, b).detach().numpy()

    assert np.allclose(euclidean_expected, euclidean_calculated)


def test_manhattan_sim() -> None:
    a = np.array([[1, 0], [0, 1]], dtype=np.float32)
    b = np.array([[0, 0], [0, 1]], dtype=np.float32)

    manhattan_expected = np.array([[-1.0, -2.0], [-1.0, 0]])
    manhattan_calculated = manhattan_sim(a, b).detach().numpy()
    assert np.allclose(manhattan_expected, manhattan_calculated)


def test_dot_score_cos_sim() -> None:
    a = np.array([[1, 0]], dtype=np.float32)
    b = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)

    dot_and_cosine_expected = np.array([[1.0, 0.0, -1.0]])
    cosine_calculated = cos_sim(a, b)
    dot_calculated = dot_score(a, b)

    assert np.allclose(cosine_calculated, dot_and_cosine_expected)
    assert np.allclose(dot_calculated, dot_and_cosine_expected)


def create_sparse_tensor(rows, cols, num_nonzero, seed=None):
    """Create a sparse tensor of shape (rows, cols) with num_nonzero values per row."""
    if seed is not None:
        torch.manual_seed(seed)

    indices = []
    values = []

    for i in range(rows):
        row_indices = torch.stack(
            [torch.full((num_nonzero,), i, dtype=torch.long), torch.randint(0, cols, (num_nonzero,))]
        )
        row_values = torch.randn(num_nonzero)

        indices.append(row_indices)
        values.append(row_values)

    indices = torch.cat(indices, dim=1)
    values = torch.cat(values)
    return torch.sparse_coo_tensor(indices, values, (rows, cols)).coalesce()


@pytest.fixture
def sparse_tensors():
    """Create two large sparse tensors of shape (50, 100) each."""
    rows, cols = 50, 1000
    num_nonzero = 10  # per row

    tensor1 = create_sparse_tensor(rows, cols, num_nonzero, seed=42)
    tensor2 = create_sparse_tensor(rows, cols, num_nonzero, seed=1337)
    if torch.cuda.is_available():
        return tensor1.to("cuda"), tensor2.to("cuda")
    else:
        return tensor1, tensor2


def test_cos_sim_sparse(sparse_tensors):
    """Test cosine similarity between sparse and dense representations."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = cos_sim(tensor1, tensor2)
    sim_dense = cos_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_dot_score_sparse(sparse_tensors):
    """Test dot product with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    # Convert to dense before computing
    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    score_sparse = dot_score(tensor1, tensor2)
    score_dense = dot_score(dense1, dense2)

    assert torch.allclose(score_sparse, score_dense, rtol=1e-5, atol=1e-5)


def test_manhattan_sim_sparse(sparse_tensors):
    """Test Manhattan similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = manhattan_sim(tensor1, tensor2)
    sim_dense = manhattan_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_euclidean_sim_sparse(sparse_tensors):
    """Test Euclidean similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = euclidean_sim(tensor1, tensor2)
    sim_dense = euclidean_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_pairwise_cos_sim_sparse(sparse_tensors):
    """Test pairwise cosine similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = pairwise_cos_sim(tensor1, tensor2)
    sim_dense = pairwise_cos_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_pairwise_dot_score_sparse(sparse_tensors):
    """Test pairwise dot product with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    score_sparse = pairwise_dot_score(tensor1, tensor2)
    score_dense = pairwise_dot_score(dense1, dense2)

    assert torch.allclose(score_sparse, score_dense, rtol=1e-5, atol=1e-5)


def test_pairwise_manhattan_sim_sparse(sparse_tensors):
    """Test pairwise Manhattan similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = pairwise_manhattan_sim(tensor1, tensor2)
    sim_dense = pairwise_manhattan_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_pairwise_euclidean_sim_sparse(sparse_tensors):
    """Test pairwise Euclidean similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = pairwise_euclidean_sim(tensor1, tensor2)
    sim_dense = pairwise_euclidean_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_maxsim_ragged_matrix_hand_computed() -> None:
    """MaxSim of ragged (variable-length) inputs, with a hand-computed (2, 2) score matrix."""
    queries = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),  # 2 tokens
        torch.tensor([[1.0, 1.0]]),  # 1 token
    ]
    documents = [
        torch.tensor([[1.0, 0.0], [0.0, 0.5]]),  # 2 tokens
        torch.tensor([[0.0, 1.0]]),  # 1 token
    ]
    # res[i][j] = sum over query tokens of the max similarity to any document token, e.g.
    # res[0][0] = max(1, 0) + max(0, 0.5) = 1.5. res[0][1] = max(0) + max(1) = 1.0
    expected = torch.tensor([[1.5, 1.0], [1.0, 1.0]])
    scores = maxsim(queries, documents)
    assert scores.shape == (2, 2)
    assert torch.allclose(scores, expected)


def test_maxsim_list_matches_masked_padded_tensor() -> None:
    """The ragged-list path auto-derives a length mask. It must match an explicitly padded + masked 3D tensor."""
    queries = [torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([[1.0, 1.0]])]
    documents = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0], [0.5, 0.5]])]
    from_list = maxsim(queries, documents)

    queries_padded = torch.nn.utils.rnn.pad_sequence(queries, batch_first=True)
    documents_padded = torch.nn.utils.rnn.pad_sequence(documents, batch_first=True)
    query_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    document_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    from_padded = maxsim(queries_padded, documents_padded, a_mask=query_mask, b_mask=document_mask)

    assert torch.allclose(from_list, from_padded)


def test_maxsim_document_mask_excludes_padding_tokens() -> None:
    """A masked-out document token must not participate in the max, even with the highest similarity."""
    query = torch.tensor([[[1.0, 0.0]]])  # (1 document-query, 1 token, dim 2)
    document_real_only = torch.tensor([[[0.8, 0.6]]])  # similarity 0.8
    document_with_pad = torch.tensor([[[0.8, 0.6], [5.0, 0.0]]])  # 2nd token would otherwise dominate the max
    mask = torch.tensor([[1.0, 0.0]])

    base = maxsim(query, document_real_only)
    masked = maxsim(query, document_with_pad, b_mask=mask)
    assert torch.allclose(base, masked)
    # Without the mask the high-similarity padding token wins, inflating the score.
    assert maxsim(query, document_with_pad).item() > base.item()


def test_maxsim_negative_similarities_survive_padding() -> None:
    """A padding token (from a shorter document) must not inflate a negative MaxSim score to 0."""
    query = [torch.tensor([[1.0, 0.0]])]  # 1 query, 1 token
    documents = [
        torch.tensor([[-0.5, 0.0]]),  # 1 token: best (only) similarity -0.5
        torch.tensor([[-0.5, 0.0], [-0.3, 0.0]]),  # 2 tokens: best similarity -0.3
    ]
    # The first document is padded to length 2. The padding must be excluded from the max so the score
    # stays -0.5 instead of the 0 a zero-valued padding token would otherwise produce.
    scores = maxsim(query, documents)
    assert torch.allclose(scores, torch.tensor([[-0.5, -0.3]]))


def test_maxsim_query_mask_excludes_query_tokens() -> None:
    """MaxSim sums over query tokens. Masking a query token drops its contribution from the sum."""
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # 2 query tokens
    document = torch.tensor([[[1.0, 0.0]]])  # token0 similarity 1, token1 similarity 0
    assert torch.allclose(maxsim(query, document), torch.tensor([[1.0]]))
    mask = torch.tensor([[0.0, 1.0]])  # keep only the second (zero-similarity) query token
    assert torch.allclose(maxsim(query, document, a_mask=mask), torch.tensor([[0.0]]))


def test_maxsim_pairwise_matches_maxsim_diagonal() -> None:
    """maxsim_pairwise(q, d) must equal the diagonal of the full maxsim(q, d) matrix, even with padding."""
    # randn gives mixed-sign similarities and the documents have different lengths, so the padded maxsim
    # path and the unpadded pairwise path only agree if masked padding tokens are excluded from the max.
    torch.manual_seed(0)
    queries = [torch.randn(3, 8), torch.randn(2, 8)]
    documents = [torch.randn(4, 8), torch.randn(6, 8)]
    full = maxsim(queries, documents)
    pairwise = maxsim_pairwise(queries, documents)
    assert pairwise.shape == (2,)
    assert torch.allclose(pairwise, torch.diagonal(full), atol=1e-5)


def test_pairwise_angle_sim_even_and_odd_sparse_embeddings(splade_bert_tiny_model: SparseEncoder) -> None:
    """Ensure pairwise_angle_sim works for even and artificially odd dims."""

    model = splade_bert_tiny_model
    sentences = [
        "The weather is nice today.",
        "It's sunny outside.",
        "I love going for walks in the park.",
        "Let's have a picnic this weekend.",
    ]
    embeddings_even_dense = model.encode(
        sentences,
        convert_to_tensor=True,
        convert_to_sparse_tensor=False,
    )
    assert embeddings_even_dense.shape[1] % 2 == 0, "Test setup error: expected even-dimensional sparse embeddings."

    # Baseline with the model's normal (even-dimensional) sparse embeddings
    sim_even = pairwise_angle_sim(embeddings_even_dense[:2].to_sparse(), embeddings_even_dense[2:].to_sparse())

    # Convert to dense, append a trailing zero column to make the embedding
    # dimension odd while keeping the representation identical, then convert
    # back to sparse so we continue testing the sparse input path.
    embeddings_odd = torch.nn.functional.pad(embeddings_even_dense, (0, 1), mode="constant", value=0)
    assert embeddings_odd.shape[1] % 2 == 1, "Test setup error: expected odd-dimensional sparse embeddings."

    sim_odd = pairwise_angle_sim(embeddings_odd[:2].to_sparse(), embeddings_odd[2:].to_sparse())

    assert sim_even.shape == sim_odd.shape
    assert torch.allclose(sim_even, sim_odd, rtol=1e-5, atol=1e-5)


def test_maxsim_document_chunking_matches_unchunked() -> None:
    """The chunked path masks and sums per chunk (bounding peak memory), which must be numerically
    identical to the single-einsum path, including with query and document masks."""
    generator = torch.Generator().manual_seed(7)
    a = torch.nn.functional.normalize(torch.randn(5, 9, 16, generator=generator), p=2, dim=-1)
    b = torch.nn.functional.normalize(torch.randn(13, 11, 16, generator=generator), p=2, dim=-1)
    a_mask = torch.rand(5, 9, generator=generator) > 0.2
    b_mask = torch.rand(13, 11, generator=generator) > 0.2
    a_mask[:, 0] = True
    b_mask[:, 0] = True

    unchunked = maxsim(a, b, a_mask=a_mask, b_mask=b_mask)
    # Budgets picked to yield roughly 1, 4, and 13 documents per chunk (per-document cost 5*9*11).
    for budget in (1, 2000, 6500, 10**12):
        chunked = maxsim(a, b, a_mask=a_mask, b_mask=b_mask, document_chunk_elements=budget)
        assert torch.allclose(unchunked, chunked, atol=1e-6), f"budget={budget}"


def test_maxsim_list_documents_chunking_pads_per_chunk() -> None:
    """Ragged document lists are padded per chunk, so one long outlier document no longer sizes the
    padding of every chunk (the multi-GB-allocation failure mode on large corpora). Must stay
    numerically identical to the unchunked globally-padded path."""
    generator = torch.Generator().manual_seed(11)
    a = torch.nn.functional.normalize(torch.randn(3, 6, 8, generator=generator), p=2, dim=-1)
    lengths = [2, 40, 3, 5, 1, 7]
    b = [torch.nn.functional.normalize(torch.randn(length, 8, generator=generator), p=2, dim=-1) for length in lengths]

    unchunked = maxsim(a, b)
    # Budgets small enough to split the ragged corpus into several chunks (per-document cost 3*6*width).
    for budget in (1, 400, 1200):
        chunked = maxsim(a, b, document_chunk_elements=budget)
        assert torch.allclose(unchunked, chunked, atol=1e-6), f"budget={budget}"

    # A caller-provided mask spans the global width and gets truncated to each chunk's local width.
    b_mask = torch.zeros(len(lengths), max(lengths), dtype=torch.bool)
    for row, length in enumerate(lengths):
        b_mask[row, :length] = True
    b_mask[1, 20:] = False
    unchunked = maxsim(a, b, b_mask=b_mask)
    for budget in (1, 400, 1200):
        chunked = maxsim(a, b, b_mask=b_mask, document_chunk_elements=budget)
        assert torch.allclose(unchunked, chunked, atol=1e-6), f"budget={budget}"


def test_maxsim_singular_inputs_score_as_batch_of_one() -> None:
    """Mirrors the dense family's similarity, where a singular (unbatched) embedding auto-batches:
    a bare 2D (tokens, dim) input, e.g. from a singular encode call, scores as a batch of one."""
    query = torch.randn(5, 8)
    documents = [torch.randn(7, 8), torch.randn(3, 8)]
    batched = maxsim([query], documents)

    assert batched.shape == (1, 2)
    assert torch.equal(maxsim(query, documents), batched)
    assert torch.equal(maxsim(documents, query), maxsim(documents, [query]))
    assert maxsim(query, query).shape == (1, 1)
    # The encode() default output type for a singular input is a 2D numpy array.
    numpy_scores = maxsim(query.numpy(), [document.numpy() for document in documents])
    assert torch.allclose(numpy_scores, batched)


def test_maxsim_pairwise_singular_inputs_score_as_batch_of_one() -> None:
    """maxsim_pairwise must auto-batch singular inputs just like maxsim, so a user who reaches for
    similarity_pairwise on singular encode outputs does not hit the einsum crash instead."""
    query = torch.randn(5, 8)
    document = torch.randn(7, 8)
    batched = maxsim_pairwise([query], [document])

    assert batched.shape == (1,)
    assert torch.allclose(maxsim_pairwise(query, document), batched, atol=1e-6)
    assert torch.allclose(maxsim_pairwise(query.numpy(), document.numpy()), batched, atol=1e-6)

    a_mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    b_mask = torch.tensor([1, 1, 1, 1, 0, 0, 0])
    assert torch.allclose(
        maxsim_pairwise(query, document, a_mask=a_mask, b_mask=b_mask),
        maxsim_pairwise([query], [document], a_mask=a_mask.unsqueeze(0), b_mask=b_mask.unsqueeze(0)),
        atol=1e-6,
    )


def test_maxsim_pairwise_derives_padding_mask_from_zero_rows() -> None:
    """Pre-padded 3D input without a mask must have its zero rows excluded from the max, both in the
    per-pair loop (mixed list + 3D, whose 2D slices keep their padding rows) and in the tensor branch.
    Visible with all-negative similarities, where a padding zero would beat every real token."""
    queries = [-torch.rand(5, 4) - 0.1, -torch.rand(3, 4) - 0.1]
    documents = [torch.rand(6, 4) + 0.1, torch.rand(2, 4) + 0.1]
    reference = maxsim_pairwise(queries, documents)
    assert (reference < 0).all()

    padded_queries = torch.nn.utils.rnn.pad_sequence(queries, batch_first=True)
    padded_documents = torch.nn.utils.rnn.pad_sequence(documents, batch_first=True)
    assert torch.allclose(maxsim_pairwise(queries, padded_documents), reference, atol=1e-6)
    assert torch.allclose(maxsim_pairwise(padded_queries, documents), reference, atol=1e-6)
    assert torch.allclose(maxsim_pairwise(queries, padded_documents.numpy()), reference, atol=1e-6)
    assert torch.allclose(maxsim_pairwise(padded_queries, padded_documents), reference, atol=1e-6)


def test_maxsim_integer_embeddings_upcast() -> None:
    """Quantized integer embeddings (encode(precision="int8")) are upcast to float so einsum and the
    finfo-based masking work, and the scores match float inputs with the same values."""
    queries = [torch.tensor([[3, -2], [0, 1]], dtype=torch.int8), torch.tensor([[1, 1]], dtype=torch.int8)]
    documents = [torch.tensor([[2, 0], [0, 5]], dtype=torch.int8), torch.tensor([[0, 1]], dtype=torch.int8)]
    expected = maxsim([q.float() for q in queries], [d.float() for d in documents])

    scores = maxsim(queries, documents)
    assert scores.dtype == torch.float32
    assert torch.allclose(scores, expected)

    queries_padded = torch.nn.utils.rnn.pad_sequence(queries, batch_first=True)
    documents_padded = torch.nn.utils.rnn.pad_sequence(documents, batch_first=True)
    padded_scores = maxsim(queries_padded, documents_padded)
    assert padded_scores.dtype == torch.float32
    assert torch.allclose(padded_scores, expected)

    pairwise = maxsim_pairwise(queries, documents)
    assert pairwise.dtype == torch.float32
    assert torch.allclose(pairwise, torch.diagonal(expected))

    pairwise_padded = maxsim_pairwise(queries_padded, documents_padded)
    assert pairwise_padded.dtype == torch.float32
    assert torch.allclose(pairwise_padded, torch.diagonal(expected))


def test_document_chunk_ranges_budget_packing() -> None:
    """Greedy packing keeps each chunk's num_queries * query_tokens * width * count under the
    budget, isolates a long outlier in its own chunk, and never emits an empty chunk."""
    # per_document_token = 2 * 8 = 16. Budget 256 allows width * count <= 16.
    ranges = _document_chunk_ranges([4, 4, 100, 4], num_queries=2, query_tokens=8, element_budget=256)
    assert ranges == [(0, 2), (2, 3), (3, 4)]

    # A huge budget packs everything into one chunk, the unchunked-equivalent case.
    assert _document_chunk_ranges([4, 4, 100, 4], 2, 8, 10**12) == [(0, 4)]

    # The one-document floor: a single document over budget still gets a chunk.
    assert _document_chunk_ranges([1000], 2, 8, 256) == [(0, 1)]

    # Uniform widths degrade to fixed-size-style chunks: each 2-document chunk costs exactly 200.
    assert _document_chunk_ranges([10] * 6, 1, 10, 200) == [(0, 2), (2, 4), (4, 6)]


def test_maxsim_element_budget_matches_single_chunk() -> None:
    """A tiny element budget (many chunks, outlier isolated) must score identically to a huge
    budget (one chunk), for ragged lists, with a caller mask, and for pre-padded tensors."""
    generator = torch.Generator().manual_seed(13)
    queries = [torch.randn(4, 8, generator=generator), torch.randn(6, 8, generator=generator)]
    lengths = [3, 40, 5, 9, 2, 12]
    documents = [torch.randn(length, 8, generator=generator) for length in lengths]

    single_chunk = maxsim(queries, documents, document_chunk_elements=10**12)
    tiny_budget = maxsim(queries, documents, document_chunk_elements=1)
    assert torch.allclose(single_chunk, tiny_budget, atol=1e-6)
    default_budget = maxsim(queries, documents)
    assert torch.allclose(single_chunk, default_budget, atol=1e-6)

    b_mask = torch.zeros(len(lengths), max(lengths), dtype=torch.bool)
    for row, length in enumerate(lengths):
        b_mask[row, :length] = True
    b_mask[1, 20:] = False
    masked_single = maxsim(queries, documents, b_mask=b_mask, document_chunk_elements=10**12)
    masked_tiny = maxsim(queries, documents, b_mask=b_mask, document_chunk_elements=1)
    assert torch.allclose(masked_single, masked_tiny, atol=1e-6)

    documents_padded = torch.nn.utils.rnn.pad_sequence(documents, batch_first=True)
    padded_single = maxsim(queries, documents_padded, b_mask=b_mask, document_chunk_elements=10**12)
    padded_tiny = maxsim(queries, documents_padded, b_mask=b_mask, document_chunk_elements=1)
    assert torch.allclose(padded_single, padded_tiny, atol=1e-6)
    assert torch.allclose(masked_single, padded_single, atol=1e-6)


def test_maxsim_half_precision_accumulates_in_float32() -> None:
    """Summing the query-token maxima in bf16 collapses documents onto the coarse bf16 grid: the
    accumulation runs in float32 and the scores stay float32."""
    generator = torch.Generator().manual_seed(17)
    queries = torch.nn.functional.normalize(torch.randn(2, 32, 16, generator=generator), p=2, dim=-1)
    base = torch.randn(1, 20, 16, generator=generator)
    documents = torch.nn.functional.normalize(base + 0.02 * torch.randn(64, 20, 16, generator=generator), p=2, dim=-1)
    queries_bf16 = queries.bfloat16()
    documents_bf16 = documents.bfloat16()
    reference = maxsim(queries_bf16.float(), documents_bf16.float())

    scores = maxsim(queries_bf16, documents_bf16)
    assert scores.dtype == torch.float32
    # Inputs are identical, so only the bf16 einsum separates the two scorings.
    assert torch.allclose(scores, reference, atol=0.05)
    # Far more distinct scores than the bf16 grid can represent for these clustered documents.
    assert scores[0].unique().numel() > 2 * scores[0].bfloat16().unique().numel()

    chunked = maxsim(queries_bf16, documents_bf16, document_chunk_elements=1)
    assert chunked.dtype == torch.float32
    assert torch.equal(chunked, scores)


def test_maxsim_pairwise_half_precision_accumulates_in_float32() -> None:
    """Both maxsim_pairwise paths (tensor and per-pair list) accumulate in float32."""
    generator = torch.Generator().manual_seed(19)
    queries = torch.nn.functional.normalize(torch.randn(6, 32, 16, generator=generator), p=2, dim=-1).bfloat16()
    documents = torch.nn.functional.normalize(torch.randn(6, 20, 16, generator=generator), p=2, dim=-1).bfloat16()
    reference = maxsim_pairwise(queries.float(), documents.float())

    tensor_scores = maxsim_pairwise(queries, documents)
    assert tensor_scores.dtype == torch.float32
    assert torch.allclose(tensor_scores, reference, atol=0.05)

    list_scores = maxsim_pairwise(list(queries.unbind(0)), list(documents.unbind(0)))
    assert list_scores.dtype == torch.float32
    assert torch.allclose(list_scores, reference, atol=0.05)


def test_maxsim_fully_masked_document_scores_sentinel(caplog) -> None:
    """A document without any unmasked token scores the -1e9 sentinel instead of overflowing to
    -inf, other documents stay unaffected, and a one-shot warning names the likely causes."""
    generator = torch.Generator().manual_seed(23)
    queries = torch.randn(2, 4, 8, generator=generator)
    documents = torch.randn(3, 5, 8, generator=generator)
    b_mask = torch.ones(3, 5, dtype=torch.bool)
    b_mask[1] = False

    with caplog.at_level("WARNING"):
        scores = maxsim(queries, documents, b_mask=b_mask)
    assert torch.isfinite(scores).all()
    assert (scores[:, 1] == -1e9).all()
    assert scores[:, [0, 2]].abs().max() < 100
    assert any("keep_only_token_ids" in record.message for record in caplog.records)

    # A zero-width document in a ragged list is the other reachable form of the same emptiness.
    list_scores = maxsim([queries[0]], [documents[0], torch.zeros(0, 8)])
    assert torch.isfinite(list_scores).all()
    assert list_scores[0, 1] == -1e9
    assert list_scores[0, 0].abs() < 100

    # The chunked path applies the same sentinel per chunk.
    chunked = maxsim(queries, documents, b_mask=b_mask, document_chunk_elements=1)
    assert torch.equal(chunked, scores)


def test_maxsim_pairwise_empty_document_scores_sentinel(caplog) -> None:
    """Empty pair entries (zero-width, all-masked, tensor mask row) score the sentinel on both
    pairwise paths, other pairs stay finite, and it warns."""
    generator = torch.Generator().manual_seed(29)
    queries = [torch.randn(3, 8, generator=generator), torch.randn(2, 8, generator=generator)]
    documents = [torch.randn(4, 8, generator=generator), torch.zeros(0, 8)]

    with caplog.at_level("WARNING"):
        scores = maxsim_pairwise(queries, documents)
    assert torch.isfinite(scores).all()
    assert scores[1] == -1e9
    assert scores[0].abs() < 100
    assert any("skiplist_words" in record.message for record in caplog.records)

    # A full-width document whose mask is all False is the other way a list entry reads as empty.
    masked_out = maxsim_pairwise(
        queries,
        [documents[0], torch.randn(4, 8, generator=generator)],
        b_mask=[torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)],
    )
    assert masked_out[1] == -1e9
    assert masked_out[0].abs() < 100

    queries_padded = torch.nn.utils.rnn.pad_sequence(queries, batch_first=True)
    documents_tensor = torch.randn(2, 4, 8, generator=generator)
    b_mask = torch.ones(2, 4, dtype=torch.bool)
    b_mask[1] = False
    tensor_scores = maxsim_pairwise(queries_padded, documents_tensor, b_mask=b_mask)
    assert torch.isfinite(tensor_scores).all()
    assert tensor_scores[1] == -1e9
    assert tensor_scores[0].abs() < 100


def test_empty_document_score_survives_the_losses_that_consume_it() -> None:
    """The sentinel must rank last AND stay finite through loss arithmetic (scaling, softmax,
    squared margins), which the float32 minimum did not."""
    generator = torch.Generator().manual_seed(31)
    queries = torch.randn(3, 4, 8, generator=generator)
    documents = torch.randn(3, 5, 8, generator=generator)
    b_mask = torch.ones(3, 5, dtype=torch.bool)
    b_mask[1] = False
    scores = maxsim(queries, documents, b_mask=b_mask)

    for scale in (1.0, 20.0):
        for targets in ([1, 0, 2], [1, 1, 2], [1, 1, 1]):
            loss = torch.nn.functional.cross_entropy(scores * scale, torch.tensor(targets))
            assert torch.isfinite(loss), f"{scale=} {targets=}"

    # MarginMSE shape: the empty document is the negative, so its sentinel reaches the loss through a
    # subtraction that masked_fill's zeroed gradient does not shield the positive from.
    queries = torch.randn(2, 4, 8, generator=generator, requires_grad=True)
    positives = torch.randn(2, 5, 8, generator=generator, requires_grad=True)
    negatives = torch.randn(2, 5, 8, generator=generator, requires_grad=True)
    margins = maxsim_pairwise(queries, positives) - maxsim_pairwise(
        queries, negatives, b_mask=torch.zeros(2, 5, dtype=torch.bool)
    )
    loss = torch.nn.functional.mse_loss(margins, torch.tensor([1.0, 1.0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(queries.grad).all()
    assert torch.isfinite(positives.grad).all()


def test_maxsim_takes_no_data_dependent_branch() -> None:
    """The unconditional sentinel fill keeps maxsim fullgraph-capturable (and free of per-call
    device syncs), which compiling the scoring hot path relies on."""
    queries = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
    documents = torch.nn.functional.normalize(torch.randn(3, 5, 8), dim=-1)
    b_mask = torch.ones(3, 5, dtype=torch.bool)
    b_mask[1] = False

    # backend="eager" keeps this about graph capture. Lowering maxsim is a separate concern: inductor
    # rejects the in-place masked_fill_ against a broadcast mask in _maxsim_reduce_documents.
    compiled = torch.compile(maxsim, fullgraph=True, backend="eager")
    scores = compiled(queries, documents, b_mask=b_mask)
    assert torch.allclose(scores, maxsim(queries, documents, b_mask=b_mask))
    assert (scores[:, 1] == -1e9).all()


@pytest.mark.parametrize(
    "convert",
    [lambda rows: rows, lambda rows: np.array(rows, dtype=np.float32), torch.tensor],
    ids=["list", "numpy", "torch"],
)
def test_maxsim_ragged_list_entry_types(convert) -> None:
    """Ragged list entries may be plain nested lists, not only arrays or tensors. Measuring their
    widths for the element budget must not assume a .shape attribute."""
    queries = [[[1.0, 0.0], [0.0, 1.0]]]
    documents = [[[0.8, 0.6]], [[0.1, 0.2], [0.3, 0.4]]]
    scores = maxsim(queries, [convert(document) for document in documents])
    assert torch.allclose(scores, torch.tensor([[1.4, 0.7]]))
