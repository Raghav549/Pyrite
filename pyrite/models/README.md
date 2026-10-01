# Model adapters

Model adapters are intentionally separated from the runtime scheduler. An adapter should expose model blocks or experts as independently loadable units and report:

- block ID and byte size
- execution order/dependencies
- quantization format
- optional expert routing metadata
- KV-cache shape/budget requirements

Large open-weight models can therefore be stored locally in shards while the runtime keeps only a bounded working set resident.
