# LFighter grounded forensic evidence report

This report summarizes the frozen development and validation evidence for the selected LFighter case. Observed measurements, interpretations, limitations, and unsupported claims are kept separate.

## Observed facts

* The plain attack mean DDoS to Benign rate was 0.191026 versus 0.056499 in the paired clean branch. [src:task45_condition]
* Mean and minimum malicious recall were both 1.000; maximum benign false positive rate was 0.100. [src:task45_condition]
* Normalized attribution distance decreased from 0.860237 to 0.520150, a recovery fraction of 0.395341. [src:task57_triplets]
* The case was selected by the frozen Task 58 median based rule without manual substitution. [src:task58_case, src:task58_decision]

## Interpretation

* The observed behavior is consistent with targeted suppression of DDoS predictions toward the Benign class, but it does not establish malicious intent. Confidence: moderate. [src:task45_condition, src:task45_rounds]
* Reconstruction moved this attribution profile toward its paired preattack clean reference; this is not evidence of oracle clean restoration. Confidence: high. [src:task57_triplets, src:task57_decision]

## Uncertainty

* The clean attribution comparator is the paired round 4 preattack reference rather than a round 8 oracle clean state.
* Rejected and oracle clean attribution states were unavailable and were not substituted.
* Per client detector scores and numeric threshold values were not present in the selected summary artifacts.
* The evidence is development and validation evidence; reserved test arrays remained closed.

## Unsupported claims refused

* Not supported by this evidence: Verified malicious intent.
* Not supported by this evidence: Exact oracle clean recovery.
* Not supported by this evidence: Generalization to unseen organizations or physical clients.
* Not supported by this evidence: Authority for an LLM to change any defense decision.

## Authority

Informational report only; the deterministic LFighter pipeline retains all security decision authority.
