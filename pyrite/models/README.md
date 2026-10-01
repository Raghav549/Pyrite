# Model adapters

Model adapters expose independently loadable layers, shared blocks, or experts.

An adapter should report block id, byte size, execution order, dependencies, quantization format, and optional routing metadata.

This allows large open-weight models to remain on local storage while Pyrite maintains a bounded in-memory working set.
