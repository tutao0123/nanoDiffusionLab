# Experimental target configurations

Files in this directory describe future experiment targets. They are intentionally excluded from
the runnable configuration set at `configs/*.py`.

These targets may require a dataset preparation recipe, tokenizer contract, or additional
validation that is not included yet. They are not covered by the CPU smoke test or continuous
integration. Treat them as documented architecture and hardware starting points, not as
reproducible commands.

The configurations become runnable only after their missing data contract is implemented and
tested.
