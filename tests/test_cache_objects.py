from pyrite.cache_objects import KVObjectCodec, ReusableKVStore


def test_quantized_kv_roundtrip_shape():
    values = [-1.0, -0.2, 0.0, 0.4, 1.0]
    payload = KVObjectCodec.encode(values, 8)
    restored = KVObjectCodec.decode(payload, 8, 1.0)
    assert len(restored) == len(values)


def test_reusable_kv_is_checksum_verified():
    store = ReusableKVStore()
    key = store.make_key("model", "tokenizer", [1, 2, 3])
    obj = store.put(key, 3, 8, b"abc")
    assert store.get(key) == obj
    assert store.verify(obj)
