Creating Custom Models
======================

Modular Architecture
--------------------

A MultiVectorEncoder consists of modules that are executed sequentially, just like the other model
families (see `Sentence Transformers > Creating Custom Models <../../sentence_transformer/usage/custom_models.html>`_
for the general module API, the saving format, and distributing custom modules via the Hub). The
default multi-vector pipeline is:

* :class:`~sentence_transformers.base.modules.Transformer`: processes the input and produces contextualized
  token embeddings. The multi-vector knobs (``query_length``, ``document_length``, ``query_expansion``) live
  here, and all start unset: inputs are capped only by the tokenizer's own maximum and queries are not
  expanded. ColBERT-style query expansion (``query_expansion={"strategy": "min", "length": 32}``, padding
  queries out to 32 ``[MASK]`` tokens, with ``"min"`` letting longer queries pass through where the classic
  ``"fixed"`` strategy would truncate them) is an explicit recipe choice, as in the MS MARCO examples.
* :class:`~sentence_transformers.base.modules.Dense` (token-level): projects each token embedding down to the
  multi-vector dimension, via ``module_input_name="token_embeddings"``. Defaults to 128 output dimensions
  with no bias and no activation, the classic ColBERT projection. Randomly initialized unless the checkpoint
  ships one, so a fresh model needs training before it is useful.
* :class:`~sentence_transformers.multi_vector_encoder.modules.MultiVectorMask`: overwrites ``attention_mask``
  with the per-row scoring mask (force-including query expansion positions, excluding document skiplist
  tokens). The skiplist starts empty, so every real token is scored. Legacy PyLate and Stanford-NLP
  checkpoints pre-seed it with punctuation instead.
* :class:`~sentence_transformers.sentence_transformer.modules.Normalize` (token-level): L2-normalizes each
  token embedding, so each MaxSim term is a cosine similarity. No configuration.

Beyond the modules, a fresh model scores with MaxSim and carries empty ``prompts``. The classic ColBERT
recipe also expands queries to 32 tokens with ``[MASK]`` padding, prepends "[Q] " and "[D] " marker tokens,
caps document length, and skips punctuation tokens during document scoring, which released checkpoints
configure themselves but a bare backbone does not.
To reproduce the full classic recipe, initialize the modules explicitly::

    import string

    from torch import nn

    from sentence_transformers import MultiVectorEncoder
    from sentence_transformers.base.modules import Dense, Transformer
    from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask
    from sentence_transformers.base.modules import Normalize

    transformer = Transformer(
        "answerdotai/ModernBERT-base",
        query_expansion={"strategy": "fixed", "length": 32},  # pad queries to 32 tokens with [MASK], truncate longer ones
        document_length=300,  # also truncate (not pad) documents to 300 tokens
    )
    dense = Dense(
        in_features=transformer.get_embedding_dimension(),
        out_features=128,
        bias=False,
        activation_function=nn.Identity(),
        module_input_name="token_embeddings",
    )
    mask = MultiVectorMask(skiplist_words=list(string.punctuation))  # skip punctuation for queries
    normalize = Normalize(module_input_name="token_embeddings")

    model = MultiVectorEncoder(
        modules=[transformer, dense, mask, normalize],
        prompts={"query": "[Q] ", "document": "[D] "},
    )

An optional :class:`~sentence_transformers.multi_vector_encoder.modules.HierarchicalTokenPooling`
module can be appended after ``Normalize`` to bake document token pooling into the checkpoint.

Extra per-token features
------------------------

Every consumer in the pipeline (``encode``, the losses, gradient caching, ``model.similarity``, and
any vector index) transports exactly one per-token tensor, ``token_embeddings``, plus its
``attention_mask``. Custom modules that produce additional per-token scalars (e.g. learned token
weights or salience scores) should therefore append them as trailing columns of
``token_embeddings`` rather than as separate feature keys: a trailing column flows through encoding,
gradient caching, padding, multi-GPU gathering, and index storage without any further wiring.

Conventions for a feature-column module:

- Place the module after ``Normalize`` in the pipeline, so the L2 normalization never sees the
  extra column.
- Pair it with a scoring function that splits the column back off (e.g. slice ``[..., :-1]`` for the
  embeddings and ``[..., -1]`` for the weights) and pass it as ``score_metric`` to the losses. The
  default MaxSim functions are channel-blind and would fold the extra column into the dot products.
- Note that ``get_embedding_dimension()`` counts the extra columns, and ``precision`` quantization
  applies to them.

For ad-hoc access to *named* module outputs (without the trailing-column convention), pass
``output_value=None`` to the encode methods to get the raw per-input feature dicts::

    outputs = model.encode_query(queries, output_value=None)
    print(outputs[0].keys())  # dict_keys(['input_ids', 'attention_mask', 'token_embeddings', ...])

Named keys are reachable this way for inspection, but they do not flow through ``model.similarity``
or the cached losses: features that must reach scoring belong in trailing columns.
