Run tests by using the command

```bash
pytest tests
```

| #  | Function                              | Implemented                                                                                                                                                                      | Test |
|----|:--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| 1  | `binContinuousCovariate`              | Yes                                                                                                                                                                              | Yes  |
| 2  | `calculateFactorLevelCut`             | Yes                                                                                                                                                                              | Yes  |
| 3  | `AssignKeyColumns`                    | Yes                                                                                                                                                                              | Yes  |
| 4  | `computePvalueInteractionWithAssetID` | Yes                                                                                                                                                                              | Yes  |
| 5  | `collapseFactorData`                  | Yes                                                                                                                                                                              | Yes  |
| 6  | `covToFactorData`                     | `sapply` function in `r` takes table as an argument, and doesn't make sense with the result`sapply` function with `covVector$faultDurationsTime` agrees with the `pandas` result | No   |
| 7  | `completeCollapsedData`               | Yes                                                                                                                                                                              | Yes  |
| 8  | `fitRegressionModelFast`              | Error running the `r` script                                                                                                                                                     | No   |
| 9  | `calculatePvalueAppended`             | Depends: `fitRegressionModelFast`                                                                                                                                                | No   |
| 10 | `significanceTestPoisModel`           | Depends: `fitRegressionModelFast`                                                                                                                                                | No   |
| 11 | `computePvalueForVar`                 | Depends: `significanceTestPoisModel`                                                                                                                                             | No   |
| 12 | `calculateRatioSE`                    | `scipy.stats.bootstrap` not yielding expected results                                                                                                                            | No   |
| 13 | `robustCut`                           | Yes                                                                                                                                                                              | Yes  |
| 14 | `removeOutliersQuantile`              | Yes                                                                                                                                                                              | Yes  |
| 15 | `removeOutliersGLMfast`               | Error running the `r` script                                                                                                                                                     | No   |
| 16 | `compute_pValues_af`                  | Depends: `significanceTestPoisModel`                                                                                                                                             | No   |
| 17 | `glm_mean_test`                       | Undefined: `quassipoisson`                                                                                                                                                       | No   |
