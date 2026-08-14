# LFighter grounded forensic evidence report

This deterministic report summarizes the frozen development and validation evidence. Facts, interpretations, uncertainty, and unsupported claims remain separate.

## Observed facts

* The plain attack mean DDoS to Benign prediction rate was 0.329128 versus 0.040246 in the paired clean branch. [src:task45_condition]
* Mean malicious recall was 0.975000; minimum malicious recall was 0.900000; maximum benign false positive rate was 0.000000. [src:task45_condition]
* Normalized attribution distance decreased from 0.753257 to 0.467583, with recovery fraction 0.379252. [src:task57_triplets]
* The defended mean DDoS to Benign prediction rate was 0.042709, an absolute reduction of 0.286420 from the plain attack branch. [src:task45_condition]

## Interpretation

* The observed prediction behavior is consistent with targeted suppression of DDoS predictions toward the Benign class, but it does not establish malicious intent. Confidence: moderate. [src:task45_condition, src:task45_rounds]
* Reconstruction moved the attribution profile toward its paired preattack clean reference; this is not evidence of oracle clean restoration. Confidence: high. [src:task57_triplets, src:task57_decision]

## Uncertainty

* The clean attribution comparator is the paired round 4 preattack reference rather than an oracle clean state.
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
