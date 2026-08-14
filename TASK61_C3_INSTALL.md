# Task 61 C3 post hoc failure diagnostic

This package diagnoses the frozen Task 61 C2 automated failure without making any API call, changing any threshold, selecting a preferred output, training a model, evaluating SHAP, or opening reserved test arrays.

Run from the repository root:

```bat
call scripts\run_task61_c3_v424.bat
```

The diagnostic reproduces the parent numeric consistency result, separates scientific measurement selection from contextual numeric selection, audits the generated evidence, and retains the original FAIL result.
