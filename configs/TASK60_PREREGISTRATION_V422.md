# Task 60 C0 preregistration

Task 60 implements the frozen deterministic and LLM forensic report generators required by roadmap item 60. It does not evaluate superiority, factuality, analyst usefulness, or prompt injection resistance. Those remain Tasks 61 and 62.

The sole scientific input is the validated Task 59 canonical evidence record. The deterministic baseline and the LLM use the same compact evidence payload and the same report interface.

The primary LLM is `gpt-5.6-terra` through the OpenAI Responses API. Reasoning effort is `low`, output is capped at 700 tokens, retries are disabled, tools and conversation history are absent, and exactly one API call is permitted for the canonical record. The zero cost deterministic template is always generated first.

The output must separate observed facts, interpretations, uncertainty, and refusals. Every factual or interpretive item must cite source identifiers already present in Task 59. Unsupported claims must be refused. The report is informational only and cannot flag clients, change thresholds or trust weights, reconstruct updates, control aggregation, or assert malicious intent.

No training, new SHAP evaluation, or reserved test array access is permitted. No API key may be written to disk or printed. A rerun reuses a completed report unless the operator explicitly requests an overwrite.
