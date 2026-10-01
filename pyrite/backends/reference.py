from __future__ import annotations


class ReferenceBackend:
    def generate(self, input_ids: list[int], max_new_tokens: int, temperature: float, top_p: float) -> list[int]:
        del temperature, top_p
        generated = list(input_ids)
        seed = input_ids[-1] if input_ids else 0
        for step in range(max(0, max_new_tokens)):
            generated.append((seed + step + 1) % 65536)
        return generated
