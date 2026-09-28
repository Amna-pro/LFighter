# Coordinate Median Matched Full Grid Plan v4.32.9

This stage scales the already verified v4.32.8 coordinate median comparator from one attacked preflight to the complete frozen reviewer grid.

Grid: five final model seeds by five attacks, 25 attacked conditions, four continuation rounds per condition (global rounds 5 through 8). The already audited all_to_one_benign seed 1379954285 preflight is reused; the other 24 conditions are newly executed.

The scientific implementation is unchanged. Every condition begins from its corresponding frozen clean round 4 warmup checkpoint and reuses the exact recovered Task 65 plain FedAvg poison plan for that attack and seed. Local model architecture, optimizer, batch size, learning rate, weight decay, class weighting, gradient clipping, and deterministic per client round seed remain matched.

The only baseline operation is: all 20 submitted floating client updates -> coordinate wise median -> next global state. There is no detector, rejection, reconstruction, sample count weighting inside the coordinate median, or attack specific tuning.

Training loads train and validation arrays only. Final diagnostic and natural test arrays are not materialized. Validation outcomes are descriptive only and cannot be used to remove attacks, remove seeds, or retune the comparator. No formal cross method statistical test or final test evaluation is performed in this stage.
