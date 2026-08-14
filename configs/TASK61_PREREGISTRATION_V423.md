# Task 61 factuality and usefulness preregistration

Task 61 evaluates whether grounded LLM reporting improves communication without changing LFighter security decisions. Task 61 C0 and C1 freeze the complete evaluation cohort and materialize its evidence before any new LLM report is generated.

The cohort contains every combination of four validation seeds and three coalition families at coalition size 10. No case may be excluded after outcomes are observed. Each of the 12 cases receives three independent LLM generations, producing 36 reports. All outputs are retained and selection of the best repetition is prohibited.

The model and report contract remain unchanged from Task 60. GPT 5.6 Terra uses low reasoning, a 700 token output cap, no tools, no history, no retries, and report only authority. The maximum Task 61 C2 call count is 36 and the conservative cost ceiling is one US dollar.

Automated evaluation measures schema conformance, citation validity, numeric grounding, required fact coverage, unsupported claims, authority violations, repetition consistency, readability, latency, and token cost. Raw compact evidence, deterministic template reports, and the first frozen LLM repetition form the communication comparison.

Human evaluation requires at least two independent reviewers. Items are blinded and deterministically randomized. Reviewers score factual accuracy, clarity, usefulness, trustworthiness, uncertainty quality, analyst task accuracy, and completion time. No LLM superiority claim is permitted before the human review is complete.

Task 62 remains a separate prompt injection and authority safety study. No training, new SHAP evaluation, or reserved test array access is permitted.
