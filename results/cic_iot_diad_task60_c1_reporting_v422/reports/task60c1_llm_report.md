# LFighter Evidence Summary

Development and validation evidence records elevated DDoS-to-Benign attribution in the plain attack branch relative to the paired clean branch, with reduced attribution distance after reconstruction. These observations do not establish malicious intent or oracle clean recovery.

## Observed facts

* The plain attack mean DDoS-to-Benign rate was 0.191026, versus 0.056499 in the paired clean branch. [src:task45_condition]
* Mean and minimum malicious recall were 1.000, and maximum benign false-positive rate was 0.100. [src:task45_condition]
* Normalized attribution distance decreased from 0.860237 to 0.520150; the reported recovery fraction was 0.395341. [src:task57_triplets]
* The case was selected by the frozen Task 58 median-based rule without manual substitution. [src:task58_case, src:task58_decision]

## Interpretation

* Observed behavior is consistent with targeted suppression of DDoS predictions toward the Benign class, but does not establish malicious intent. Confidence: moderate. [src:task45_condition, src:task45_rounds]
* Reconstruction moved the attribution profile toward its paired preattack clean reference; this is not evidence of oracle clean restoration. Confidence: high. [src:task57_triplets, src:task57_decision]

## Uncertainty

* The clean attribution comparator was the paired round-4 preattack reference, not a round-8 oracle clean state.
* Rejected and oracle-clean attribution states were unavailable and were not substituted.
* Per-client detector scores and numeric thresholds were absent from the selected summary artifacts.
* Evidence is limited to development and validation; reserved test arrays remained closed.

## Unsupported claims refused

* Refuse to claim verified malicious intent: the supplied evidence does not support it.
* Refuse to claim oracle clean recovery: an oracle-clean comparator was unavailable.
* No defense decision, client flag, threshold, trust-weight, reconstruction, or aggregation decision is made in this report.

## Authority

Informational report only; the deterministic LFighter pipeline retains all security decision authority.
