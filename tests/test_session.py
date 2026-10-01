from pyrite.backends.reference import ReferenceBackend
from pyrite.config import RuntimeConfig
from pyrite.engine import PyriteRuntime
from pyrite.session import LocalSession
from pyrite.tokenizer import WhitespaceTokenizer


def test_local_session():
    runtime = PyriteRuntime(RuntimeConfig(max_kv_tokens=32))
    session = LocalSession(runtime, WhitespaceTokenizer(), ReferenceBackend())
    result = session.generate("python api", max_new_tokens=2)
    assert result.route == "coding"
    assert result.tokens_in == 2
    assert result.tokens_out == 2
