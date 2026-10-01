from pyrite.backends.reference import ReferenceBackend
from pyrite.inference import LocalInference
from pyrite.tokenizer import WhitespaceTokenizer


def test_inference_contract():
    tokenizer = WhitespaceTokenizer()
    result = LocalInference(tokenizer, ReferenceBackend()).generate("hello", 2)
    assert result.startswith("[")
