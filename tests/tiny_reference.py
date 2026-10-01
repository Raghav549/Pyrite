"""Independent, deliberately naive Qwen3-MoE forward pass.

Written as a straightforward "recompute everything for the prefix" model with
row-major matrices, so it shares no code with ``pyrite.executor`` (which streams
GGUF blocks through an LRU cache and keeps an incremental KV cache).  Agreement
between the two is what the end-to-end test asserts.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

from pyrite.adapters.gguf import GGUFReader
from pyrite.qwen3_moe import EXPERT_TENSORS, Qwen3MoEConfig


class NaiveQwen3MoE:
    def __init__(self, path: str | Path):
        self.reader = GGUFReader(Path(path))
        self.config = Qwen3MoEConfig.from_metadata(self.reader.metadata())
        self.tensors: dict[str, tuple[int, ...]] = {
            tensor.name: tensor.dims for tensor in self.reader.tensor_index()
        }
        self._values: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ plumbing
    def values(self, name: str) -> list[float]:
        if name not in self._values:
            payload = self.reader.read_tensor(self.reader.tensor(name))
            if self.reader.tensor(name).ggml_type != 0:
                raise ValueError("naive reference only understands F32 tensors")
            count = len(payload) // 4
            self._values[name] = list(struct.unpack("<" + "f" * count, payload))
        return self._values[name]

    def matrix(self, name: str) -> tuple[int, int, list[tuple[float, ...]]]:
        """Return (rows, cols, rows-as-tuples) for a ggml ``(cols, rows)`` tensor."""
        cols, rows = self.tensors[name][:2]
        flat = self.values(name)
        return rows, cols, [tuple(flat[row * cols:(row + 1) * cols]) for row in range(rows)]

    # ---------------------------------------------------------------------- math
    @staticmethod
    def rms(x: list[float], weight: list[float], eps: float) -> list[float]:
        mean_square = sum(value * value for value in x) / len(x)
        scale = 1.0 / math.sqrt(mean_square + eps)
        return [value * scale * w for value, w in zip(x, weight, strict=True)]

    @staticmethod
    def linear(rows: list[tuple[float, ...]], x: list[float]) -> list[float]:
        return [sum(a * b for a, b in zip(row, x, strict=True)) for row in rows]

    @staticmethod
    def neox_rope(head: list[float], position: int, theta: float) -> list[float]:
        dim = len(head)
        half = dim // 2
        out = list(head)
        for index in range(half):
            inv = 1.0 / (theta ** (2.0 * index / dim))
            angle = position * inv
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            lo, hi = head[index], head[index + half]
            out[index] = lo * cos_a - hi * sin_a
            out[index + half] = lo * sin_a + hi * cos_a
        return out

    # -------------------------------------------------------------------- forward
    def logits(self, tokens: list[int]) -> list[float]:
        cfg = self.config
        hidden = cfg.hidden_size
        embed_rows, embed_cols, embed = self.matrix("token_embd.weight")
        assert embed_cols == hidden and embed_rows == cfg.vocab_size
        states = [list(embed[token]) for token in tokens]

        for layer in range(cfg.num_hidden_layers):
            prefix = f"blk.{layer}."
            attn_norm = self.values(prefix + "attn_norm.weight")
            q_rows, q_cols, wq = self.matrix(prefix + "attn_q.weight")
            k_rows, k_cols, wk = self.matrix(prefix + "attn_k.weight")
            v_rows, v_cols, wv = self.matrix(prefix + "attn_v.weight")
            o_rows, o_cols, wo = self.matrix(prefix + "attn_output.weight")
            q_norm = self.values(prefix + "attn_q_norm.weight")
            k_norm = self.values(prefix + "attn_k_norm.weight")
            ffn_norm = self.values(prefix + "ffn_norm.weight")
            router_rows, router_cols, router = self.matrix(prefix + "ffn_gate_inp.weight")
            assert (q_rows, q_cols, k_rows, k_cols, v_rows, v_cols) == (
                cfg.q_proj_dim, hidden, cfg.kv_proj_dim, hidden, cfg.kv_proj_dim, hidden,
            )
            assert (o_rows, o_cols) == (hidden, cfg.q_proj_dim)
            assert (router_rows, router_cols) == (cfg.num_experts, hidden)

            # Keys/values for every position first, then attention over the
            # prefix: the plain-text way to write it, matching a full prefill.
            keys: list[list[list[float]]] = []
            values: list[list[list[float]]] = []
            queries: list[list[list[float]]] = []
            for position, state in enumerate(states):
                normed = self.rms(state, attn_norm, cfg.rms_norm_eps)
                q = self.linear(wq, normed)
                k = self.linear(wk, normed)
                v = self.linear(wv, normed)
                queries.append([
                    self.neox_rope(
                        self.rms(q[h * cfg.head_dim:(h + 1) * cfg.head_dim], q_norm, cfg.rms_norm_eps),
                        position, cfg.rope_theta,
                    )
                    for h in range(cfg.num_attention_heads)
                ])
                keys.append([
                    self.neox_rope(
                        self.rms(k[h * cfg.head_dim:(h + 1) * cfg.head_dim], k_norm, cfg.rms_norm_eps),
                        position, cfg.rope_theta,
                    )
                    for h in range(cfg.num_key_value_heads)
                ])
                values.append([
                    v[h * cfg.head_dim:(h + 1) * cfg.head_dim]
                    for h in range(cfg.num_key_value_heads)
                ])

            group = cfg.num_attention_heads // cfg.num_key_value_heads
            new_states: list[list[float]] = []
            for position, state in enumerate(states):
                attended: list[float] = []
                for head, query in enumerate(queries[position]):
                    kv_index = head // group
                    scores = [
                        sum(a * b for a, b in zip(query, keys[past][kv_index], strict=True))
                        / math.sqrt(cfg.head_dim)
                        for past in range(position + 1)
                    ]
                    peak = max(scores)
                    weights = [math.exp(score - peak) for score in scores]
                    total = sum(weights)
                    weights = [value / total for value in weights]
                    head_out = [0.0] * cfg.head_dim
                    for past, weight in enumerate(weights):
                        value = values[past][kv_index]
                        for index in range(cfg.head_dim):
                            head_out[index] += weight * value[index]
                    attended.extend(head_out)
                projected = self.linear(wo, attended)
                state = [a + b for a, b in zip(state, projected, strict=True)]

                normed_ffn = self.rms(state, ffn_norm, cfg.rms_norm_eps)
                logits = self.linear(router, normed_ffn)
                peak = max(logits)
                probs = [math.exp(value - peak) for value in logits]
                total = sum(probs)
                probs = [value / total for value in probs]
                ranking = sorted(range(cfg.num_experts), key=lambda e: (-probs[e], e))
                chosen = ranking[: cfg.num_experts_per_tok]
                weights = [probs[e] for e in chosen]
                if cfg.norm_topk_prob:
                    weight_sum = sum(weights)
                    weights = [value / weight_sum for value in weights]
                if cfg.expert_weights_scale != 1.0:
                    weights = [value * cfg.expert_weights_scale for value in weights]

                moe = [0.0] * hidden
                for weight, expert in zip(weights, chosen, strict=True):
                    if weight == 0.0:
                        continue
                    gate = self._expert_matrix(prefix, expert, "gate")
                    up = self._expert_matrix(prefix, expert, "up")
                    down = self._expert_matrix(prefix, expert, "down")
                    gate_out = self.linear(gate, normed_ffn)
                    up_out = self.linear(up, normed_ffn)
                    act = [
                        (g / (1.0 + math.exp(-g))) * u
                        for g, u in zip(gate_out, up_out, strict=True)
                    ]
                    expert_out = self.linear(down, act)
                    for index in range(hidden):
                        moe[index] += weight * expert_out[index]
                new_states.append([a + b for a, b in zip(state, moe, strict=True)])
            states = new_states

        output_norm = self.values("output_norm.weight")
        final = self.rms(states[-1], output_norm, cfg.rms_norm_eps)
        out_name = "output.weight" if "output.weight" in self.tensors else "token_embd.weight"
        rows, cols, out_rows = self.matrix(out_name)
        assert cols == hidden and rows == cfg.vocab_size
        return self.linear(out_rows, final)

    def _expert_matrix(self, prefix: str, expert: int, component: str) -> list[tuple[float, ...]]:
        name = prefix + EXPERT_TENSORS[component]
        dims = self.tensors[name]
        first, second, experts = dims
        assert experts == self.config.num_experts
        flat = self.values(name)
        inner = first * second
        chunk = flat[expert * inner:(expert + 1) * inner]
        return [tuple(chunk[row * first:(row + 1) * first]) for row in range(second)]

    def greedy_tokens(self, prompt: list[int], steps: int, eos: int | None = None) -> list[int]:
        tokens = list(prompt)
        for _ in range(steps):
            logits = self.logits(tokens)
            nxt = max(range(len(logits)), key=logits.__getitem__)
            tokens.append(nxt)
            if eos is not None and nxt == eos:
                break
        return tokens
