from pyrite.inference import GenerationRequest, LocalInference


class Tokenizer:
    def encode(self, text):
        return [len(text)]

    def decode(self, tokens):
        return str(tokens)


class Backend:
    def generate(self, input_ids, max_new_tokens, temperature, top_p):
        return [*input_ids, max_new_tokens]


def test_local_inference_facade():
    result = LocalInference(Tokenizer(), Backend()).generate(
        GenerationRequest("hello", max_new_tokens=3)
    )
    assert result == "[5, 3]"
