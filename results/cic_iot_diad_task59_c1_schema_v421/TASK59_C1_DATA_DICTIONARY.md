# Task 59 forensic evidence data dictionary

The canonical record is read only input for reporting. `attack`, `detection`, `reconstruction`, `behavior`, and `xai` contain verified structured evidence. `observed_facts` contains reportable facts with source references. `interpretations` is separate and always marked as non decisive. `uncertainty` lists missing evidence and prohibited claims. `authority` is fixed by constants and grants no security control to an LLM. `provenance` maps every source identifier to a path, byte count, and SHA256 digest.

Empty score and threshold arrays mean those numeric records were not materialized in the selected summary artifacts. They must remain empty rather than being estimated or invented. The later Task 60 generator must cite source identifiers and refuse unsupported requests.
