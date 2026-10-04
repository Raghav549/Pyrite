"""Dense decoder-only checkpoints (``qwen3`` and ``llama`` architectures).

Both architectures are RMSNorm + GQA + NEOX RoPE + SwiGLU transformers; they
differ in metadata defaults and in Qwen3's per-head QK-norm.  The layout and
metadata keys follow the published GGUF contract used by llama.cpp::

    {arch}.embedding_length / block_count / feed_forward_length
    {arch}.attention.head_count / head_count_kv / key_length / value_length
    {arch}.attention.layer_norm_rms_epsilon / rope.freq_base / rope.dimension_count

:class:`DenseExecutor` reuses the streaming engine, KV cache, sampler and
reference decoders of :class:`pyrite.executor.Qwen3MoEExecutor`; only the
feed-forward block (dense SwiGLU instead of routed experts) differs.  It uses
the optional native Q4_K/Q6_K matvec kernels when they are available; the
remaining transformer operations use the Python reference path.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .adapters.gguf import GGUFReader, GGUFTensor
from .executor import Qwen3MoEExecutor
from .ggml_types import type_name
from .kernels.native import native_available, sorted_native_types
from .lm_head import (
    LMHead,
    LMHeadContractError,
    Vocabulary,
    VocabularyContractError,
    declared_tied_embeddings,
    resolve_lm_head,
    resolve_vocabulary,
)
from .tensor_ops import DECODABLE_TYPES

ARCHITECTURES = ("qwen3", "llama")

#: Published per-architecture defaults, used only when the file omits the key.
_ROPE_THETA = {"qwen3": 1_000_000.0, "llama": 500_000.0}
_RMS_EPS = {"qwen3": 1e-6, "llama": 1e-5}


class DenseContractError(ValueError):
    """Raised when a GGUF file does not satisfy the dense-model contract."""


@dataclass(frozen=True)
class DenseConfig:
    architecture: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    value_dim: int
    rotary_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    bos_token_id: int | None
    eos_token_id: int | None
    tie_word_embeddings: bool
    has_qk_norm: bool
    #: Vocabulary entries the tokenizer can name (``vocab_size`` may be padded).
    vocab_token_count: int = 0
    vocab_source: str = "unknown"
    #: ``True`` when the file itself declared tying; ``None`` means undeclared.
    tie_declared: bool | None = None

    def __post_init__(self) -> None:
        if not self.vocab_token_count:
            object.__setattr__(self, "vocab_token_count", self.vocab_size)

    @property
    def q_proj_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def v_proj_dim(self) -> int:
        return self.num_key_value_heads * self.value_dim

    @property
    def attn_output_dim(self) -> int:
        return self.num_attention_heads * self.value_dim

    @property
    def kv_cache_dim(self) -> int:
        return self.num_key_value_heads * (self.head_dim + self.value_dim)

    @property
    def kv_heads_per_group(self) -> int:
        return max(1, self.num_attention_heads // max(1, self.num_key_value_heads))

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, object],
        tensors: Mapping[str, GGUFTensor] | None = None,
    ) -> DenseConfig:
        arch = str(metadata.get("general.architecture", ""))
        if arch not in ARCHITECTURES:
            raise DenseContractError(f"unsupported GGUF architecture: {arch or 'unknown'}")

        def get(key: str, default: object = None) -> object:
            return metadata.get(key, default)

        def n(suffix: str, default: object = None) -> object:
            return get(f"{arch}.{suffix}", default)

        hidden = int(n("embedding_length", 0) or 0)
        layers = int(n("block_count", 0) or 0)
        heads = int(n("attention.head_count", 0) or 0)
        raw_kv_heads = n("attention.head_count_kv", None)
        kv_heads = heads if raw_kv_heads is None else int(raw_kv_heads)
        intermediate = int(n("feed_forward_length", 0) or 0)
        raw_head_dim = n("attention.key_length", None)
        head_dim = int(hidden // heads if raw_head_dim is None and heads else raw_head_dim or 0)
        raw_value_dim = n("attention.value_length", None)
        value_dim = int(head_dim if raw_value_dim is None else raw_value_dim)
        raw_rotary_dim = n("rope.dimension_count", None)
        rotary_dim = int(head_dim if raw_rotary_dim is None else raw_rotary_dim)

        required = {
            "embedding_length": hidden,
            "block_count": layers,
            "attention.head_count": heads,
            "attention.head_count_kv": kv_heads,
            "feed_forward_length": intermediate,
        }
        missing = sorted(name for name, value in required.items() if value <= 0)
        if missing:
            raise DenseContractError(
                f"GGUF is missing required {arch} metadata: "
                + ", ".join(f"{arch}.{name}" for name in missing)
            )
        if kv_heads > heads or heads % kv_heads:
            raise DenseContractError(
                f"attention head count {heads} must be divisible by KV head count {kv_heads}"
            )

        embedding = (tensors or {}).get("token_embd.weight")
        embedding_rows = (
            int(embedding.dims[1])
            if embedding is not None and len(embedding.dims) == 2
            else None
        )
        try:
            vocab = resolve_vocabulary(metadata, arch, embedding_rows)
        except VocabularyContractError as exc:
            raise DenseContractError(str(exc)) from exc

        raw_eps = n("attention.layer_norm_rms_epsilon", None)
        eps = float(_RMS_EPS[arch] if raw_eps is None else raw_eps)
        raw_rope_theta = n("rope.freq_base", None)
        rope_theta = float(_ROPE_THETA[arch] if raw_rope_theta is None else raw_rope_theta)
        max_pos = int(n("context_length", 0) or 0)
        bos = get("tokenizer.ggml.bos_token_id")
        eos = get("tokenizer.ggml.eos_token_id")
        declared_tied = _declared_tied(metadata, arch)
        qk_norm = _metadata_bool(n("attention.qk_norm"), arch == "qwen3", "attention.qk_norm")

        if head_dim <= 0:
            raise DenseContractError(
                f"{arch}.attention.key_length is absent and head_dim cannot be inferred "
                f"from embedding_length/attention.head_count ({hidden}/{heads})"
            )
        if value_dim <= 0 or rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
            raise DenseContractError(
                "attention value/rotary dimensions are invalid "
                f"(head_dim={head_dim}, value_length={value_dim}, "
                f"rope.dimension_count={rotary_dim})"
            )
        if not math.isfinite(eps) or eps <= 0:
            raise DenseContractError("RMSNorm epsilon must be finite and positive")
        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise DenseContractError("RoPE frequency base must be finite and positive")

        return cls(
            architecture=arch,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            value_dim=value_dim,
            rotary_dim=rotary_dim,
            vocab_size=vocab.size,
            rms_norm_eps=eps,
            rope_theta=rope_theta,
            max_position_embeddings=max_pos,
            bos_token_id=int(bos) if bos is not None else None,
            eos_token_id=int(eos) if eos is not None else None,
            # A file that says nothing about tying is tied exactly when it omits
            # output.weight; that is decided once the tensors are known, so an
            # undeclared flag stays False here and validate_contract() resolves it.
            tie_word_embeddings=bool(declared_tied) if declared_tied is not None else False,
            has_qk_norm=qk_norm,
            vocab_token_count=vocab.token_count,
            vocab_source=vocab.source,
            tie_declared=declared_tied,
        )

        arch = str(metadata.get("general.architecture", ""))
        if arch not in ARCHITECTURES:
            raise DenseContractError(f"unsupported GGUF architecture: {arch or 'unknown'}")

        def get(key: str, default: object = None) -> object:
            return metadata.get(key, default)

        def n(suffix: str, default: object = None) -> object:
            return get(f"{arch}.{suffix}", default)

        hidden = int(n("embedding_length", 0) or 0)
        layers = int(n("block_count", 0) or 0)
        heads = int(n("attention.head_count", 0) or 0)
        raw_kv_heads = n("attention.head_count_kv", None)
        kv_heads = heads if raw_kv_heads is None else int(raw_kv_heads)
        intermediate = int(n("feed_forward_length", 0) or 0)
        raw_head_dim = n("attention.key_length", None)
        head_dim = int(hidden // heads if raw_head_dim is None and heads else raw_head_dim or 0)
        raw_value_dim = n("attention.value_length", None)
        value_dim = int(head_dim if raw_value_dim is None else raw_value_dim)
        raw_rotary_dim = n("rope.dimension_count", None)
        rotary_dim = int(head_dim if raw_rotary_dim is None else raw_rotary_dim)
        vocab = _resolve_vocab_size(metadata, arch)
        raw_eps = n("attention.layer_norm_rms_epsilon", None)
        eps = float(_RMS_EPS[arch] if raw_eps is None else raw_eps)
        raw_rope_theta = n("rope.freq_base", None)
        rope_theta = float(_ROPE_THETA[arch] if raw_rope_theta is None else raw_rope_theta)
        max_pos = int(n("context_length", 0) or 0)
        bos = get("tokenizer.ggml.bos_token_id")
        eos = get("tokenizer.ggml.eos_token_id")
        tied = _metadata_bool(n("tied_word_embeddings"), False, "tied_word_embeddings")
        qk_norm = _metadata_bool(n("attention.qk_norm"), arch == "qwen3", "attention.qk_norm")

        required = (hidden, layers, heads, kv_heads, intermediate, vocab, head_dim)
        if any(value <= 0 for value in required):
            raise DenseContractError(f"GGUF is missing required {arch} metadata")
        if kv_heads > heads or heads % kv_heads:
            raise DenseContractError("attention head count must be divisible by KV head count")
        if value_dim <= 0 or rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
            raise DenseContractError("attention value/rotary dimensions are invalid")
        if not math.isfinite(eps) or eps <= 0:
            raise DenseContractError("RMSNorm epsilon must be finite and positive")
        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise DenseContractError("RoPE frequency base must be finite and positive")

        return cls(
            architecture=arch,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            value_dim=value_dim,
            rotary_dim=rotary_dim,
            vocab_size=vocab,
            rms_norm_eps=eps,
            rope_theta=rope_theta,
            max_position_embeddings=max_pos,
            bos_token_id=int(bos) if bos is not None else None,
            eos_token_id=int(eos) if eos is not None else None,
            tie_word_embeddings=tied,
            has_qk_norm=qk_norm,
        )

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def _declared_tied(metadata: Mapping[str, object], arch: str) -> bool | None:
    """Read an explicit tie declaration; ``None`` when the file makes none."""
    try:
        return declared_tied_embeddings(metadata, arch)
    except LMHeadContractError as exc:
        raise DenseContractError(str(exc)) from exc


def _metadata_bool(value: object, default: bool, name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise DenseContractError(f"{name} metadata must be a boolean")


def _resolve_vocab_size(metadata: Mapping[str, object], arch: str) -> int:
    """Metadata-only vocabulary width (kept for callers without a tensor index)."""
    try:
        return resolve_vocabulary(metadata, arch, None).size
    except VocabularyContractError as exc:
        raise DenseContractError(str(exc)) from exc


class DenseCheckpoint:
    """Inspect and validate a dense ``qwen3``/``llama`` GGUF checkpoint."""

    #: Dense tensors every layer must provide (QK-norm is optional per file).
    LAYER_TENSORS: tuple[str, ...] = (
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    )

    def __init__(self, path: str | Path, reader: GGUFReader | None = None):
        self.path = Path(path)
        self.reader = reader or GGUFReader(self.path)
        if self.reader.path != self.path:
            raise ValueError("GGUF reader belongs to a different checkpoint")
        self.metadata = self.reader.metadata()
        self._tensors = {tensor.name: tensor for tensor in self.reader.tensor_index()}
        # The tensor index is needed to resolve the vocabulary: real GGUF files
        # carry no {arch}.vocab_size key, and tokenizer-less files carry no
        # token list at all.
        self.config = DenseConfig.from_metadata(self.metadata, self._tensors)
        self._lm_head: LMHead | None = None

    @property
    def lm_head(self) -> LMHead:
        """The tensor used as the LM output projection (resolved and validated)."""
        if self._lm_head is None:
            self._lm_head = resolve_lm_head(
                self._tensors,
                self.metadata,
                self.config.architecture,
                self.config.hidden_size,
                Vocabulary(self.config.vocab_size, self.config.vocab_token_count, self.config.vocab_source),
                # Decodability is deliberately not checked here: ensure_decodable()
                # is the single gate that refuses to run on undecodable tensors, and
                # qwen3-check must be able to *report* them instead of aborting.
                decodable=None,
            )
        return self._lm_head

    @property
    def lm_head_tensor(self) -> str:
        return self.lm_head.tensor_name

    @property
    def tensor_count(self) -> int:
        return len(self._tensors)

    def tensor(self, name: str) -> GGUFTensor:
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise KeyError(f"missing tensor: {name}") from exc

    def has_tensor(self, name: str) -> bool:
        return name in self._tensors

    def validate_contract(self) -> dict[str, object]:
        """Check tensor presence and shapes; raise on any violation."""
        cfg = self.config
        problems: list[str] = []

        def require(name: str, shape: tuple[int, ...]) -> None:
            tensor = self._tensors.get(name)
            if tensor is None:
                problems.append(f"missing tensor: {name}")
                return
            if tuple(tensor.dims) != shape:
                problems.append(f"{name} has shape {tensor.dims}, expected {shape}")

        require("token_embd.weight", (cfg.hidden_size, cfg.vocab_size))
        require("output_norm.weight", (cfg.hidden_size,))
        # The LM head is resolved (not assumed): output.weight when present,
        # otherwise the token-embedding matrix, exactly as the reference
        # runtime does.  A separate output.weight is still shape-checked here so
        # a malformed one is reported alongside the other tensor problems.
        if "output.weight" in self._tensors:
            require("output.weight", (cfg.hidden_size, cfg.vocab_size))

        for layer in range(cfg.num_hidden_layers):
            prefix = f"blk.{layer}."
            require(prefix + "attn_norm.weight", (cfg.hidden_size,))
            require(prefix + "attn_q.weight", (cfg.hidden_size, cfg.q_proj_dim))
            require(prefix + "attn_k.weight", (cfg.hidden_size, cfg.kv_proj_dim))
            require(prefix + "attn_v.weight", (cfg.hidden_size, cfg.v_proj_dim))
            require(prefix + "attn_output.weight", (cfg.attn_output_dim, cfg.hidden_size))
            require(prefix + "ffn_norm.weight", (cfg.hidden_size,))
            require(prefix + "ffn_gate.weight", (cfg.hidden_size, cfg.intermediate_size))
            require(prefix + "ffn_up.weight", (cfg.hidden_size, cfg.intermediate_size))
            require(prefix + "ffn_down.weight", (cfg.intermediate_size, cfg.hidden_size))
            for norm in ("attn_q_norm.weight", "attn_k_norm.weight"):
                if cfg.has_qk_norm or prefix + norm in self._tensors:
                    require(prefix + norm, (cfg.head_dim,))

        if problems:
            raise DenseContractError(
                f"{cfg.architecture} GGUF contract violated:\n  - " + "\n  - ".join(problems[:20])
            )
        # Raises LMHeadContractError (a ValueError) when neither a separate
        # output.weight nor a legally reusable embedding matrix is available.
        try:
            head = self.lm_head
        except LMHeadContractError as exc:
            raise DenseContractError(
                f"{cfg.architecture} GGUF contract violated:\n  - {exc}"
            ) from exc
        unsupported_types = sorted(
            {
                type_name(tensor.ggml_type)
                for tensor in self._tensors.values()
                if tensor.ggml_type not in DECODABLE_TYPES
            }
        )
        return {
            "architecture": cfg.architecture,
            "layers": cfg.num_hidden_layers,
            "vocab_size": cfg.vocab_size,
            "vocab_token_count": cfg.vocab_token_count,
            "vocab_source": cfg.vocab_source,
            "lm_head": head.to_dict(),
            "unsupported_tensor_types": unsupported_types,
            "tensor_types_decodable": not unsupported_types,
            "native_generation_ready": not unsupported_types,
        }


class DenseExecutor(Qwen3MoEExecutor):
    """Real, bounded dense-transformer inference over a GGUF checkpoint."""

    CHECKPOINT_CLS = DenseCheckpoint

    #: Dense per-layer tensors prefetched ahead of execution.
    DENSE_PREFETCH_TENSORS: tuple[str, ...] = (
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "ffn_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    )

    def _prefetch_layer(self, layer: int) -> None:
        if not self.prefetch_enabled:
            return
        cfg = self.checkpoint.config
        if not 0 <= layer < cfg.num_hidden_layers:
            return
        ids = [
            f"tensor:blk.{layer}.{name}"
            for name in self.DENSE_PREFETCH_TENSORS
            if self.checkpoint.has_tensor(f"blk.{layer}.{name}")
        ]
        self.streamer.prefetch(ids)

    def _mlp(self, layer: int, hidden: Sequence[float]) -> list[float]:
        from .tensor_ops import silu

        cfg = self.checkpoint.config
        gate = self._tensor_matvec(
            f"blk.{layer}.ffn_gate.weight",
            cfg.intermediate_size,
            cfg.hidden_size,
            hidden,
        )
        up = self._tensor_matvec(
            f"blk.{layer}.ffn_up.weight",
            cfg.intermediate_size,
            cfg.hidden_size,
            hidden,
        )
        act = [s * u for s, u in zip(silu(gate), up, strict=True)]
        return self._tensor_matvec(
            f"blk.{layer}.ffn_down.weight",
            cfg.hidden_size,
            cfg.intermediate_size,
            act,
        )

    def _layer(self, layer: int, hidden: list[float], position: int) -> list[float]:
        cfg = self.checkpoint.config
        normed = self._rms_norm(hidden, self._dense_vector(f"blk.{layer}.attn_norm.weight"), cfg.rms_norm_eps)
        attention = self._attention(layer, normed, position)
        attention = self._tensor_matvec(
            f"blk.{layer}.attn_output.weight",
            cfg.hidden_size,
            cfg.attn_output_dim,
            attention,
        )
        hidden = [a + b for a, b in zip(hidden, attention, strict=True)]

        self._prefetch_layer(layer + 1)
        ffn_normed = self._rms_norm(hidden, self._dense_vector(f"blk.{layer}.ffn_norm.weight"), cfg.rms_norm_eps)
        mlp = self._mlp(layer, ffn_normed)
        self.stats.layers_executed += 1
        return [a + b for a, b in zip(hidden, mlp, strict=True)]

    def checkpoint_report(self) -> dict[str, object]:
        cfg = self.checkpoint.config
        total_bytes = sum(tensor.size for tensor in self.checkpoint.reader.tensor_index())
        return {
            "path": str(self.path),
            "architecture": cfg.architecture,
            "layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "vocab_size": cfg.vocab_size,
            "lm_head": self.checkpoint.lm_head.to_dict(),
            "vocab_token_count": cfg.vocab_token_count,
            "tensor_count": self.checkpoint.tensor_count,
            "total_bytes": total_bytes,
            "resident_budget_bytes": self.resident_bytes,
            "working_set_budget_bytes": self.runtime.config.resident_byte_budget,
            "rss_limit_bytes": self.runtime.config.ram_budget_mb * 1024 * 1024,
            "kv_budget_bytes": self.kv_budget_bytes,
            "kv_cache_policy": self.runtime.config.kv_cache_policy,
            "cpu_execution": True,
            "native_kernel_available": native_available(),
            "native_kernel_types": sorted_native_types(),
            "unsupported_tensor_types": list(self.unsupported_tensor_types()),
            "tokenizer": type(self.tokenizer).__name__,
        }
