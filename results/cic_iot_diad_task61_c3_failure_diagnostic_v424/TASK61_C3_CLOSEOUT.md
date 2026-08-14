# Task 61 C3 post hoc failure diagnostic closeout

This analysis is explicitly post hoc. It does not replace the frozen Task 61 C2 automated result, change the 0.99 threshold, select a preferred repeat, or make new API calls.

The parent repeat numeric consistency rate was independently reproduced as 0.913787383640. The frozen automated result remains FAIL.

Four cases fell below the parent threshold. Three cases changed their selection of required scientific facts across repeats: seed 123 family B, seed 2026 family A, and seed 7 family C. In each, attribution recovery numbers were omitted from at least one repeat. Seed 123 family B also varied extensive coalition and client identifier details, and one repeat split a required defense fact across two fact statements.

Seed 2026 family B retained the same required scientific measurements in all repeats. Its parent inconsistency came only from mentioning coalition size 10 in one executive summary. This is context only numeric selection, not a conflicting scientific result.

No ungrounded number was found and no repeated required fact was observed with conflicting values. The failure mechanism is variable content selection, not numeric fabrication. Restricting the diagnostic to preregistered scientific measurements raises mean consistency to 0.946296296296, but it still remains below 0.99 and is report only. It does not revise the parent gate.

The evidence directly motivates a new Task 62 protocol in which a deterministic component selects all required fact identifiers and exact formatted numbers before an LLM performs surface realization. That architecture must be evaluated prospectively and must not be presented as a repair of the frozen Task 61 result.
