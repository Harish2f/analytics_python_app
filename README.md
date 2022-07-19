Run tests by using the command

```bash
pytest tests
```

| #  | Function                              | Implemented                                          | Test |
|----|:--------------------------------------|------------------------------------------------------|------|
| 1  | `binContinuousCovariate`              | Yes                                                  | Yes  |
| 2  | `calculateFactorLevelCut`             | Yes                                                  | Yes  |
| 3  | `AssignKeyColumns`                    | Yes                                                  | Yes  |
| 4  | `computePvalueInteractionWithAssetID` | Yes                                                  | Yes  |
| 5  | `collapseFactorData`                  | `df.groupby` is not yielding expected results        | Yes  |
| 6  | `covToFactorData`                     | Depends: `collapseFactorData`                        | No   |
| 7  | `completeCollapsedData`               | Depends: `collapseFactorData`                        | No   |
| 8  | `fitRegressionModelFast`              | Error running the `r` script                           | No   |
| 9  | `calculatePvalueAppended`             | Depends: `fitRegressionModelFast`                    | No   |
| 10 | `significanceTestPoisModel`           | Depends: `fitRegressionModelFast`                    | No   |
| 11 | `computePvalueForVar`                 | Depends: `significanceTestPoisModel`                 | No   |
| 12 | `calculateRatioSE`                    | `scipy.stats.bootsrap` not yielding expected results | No   |
| 13 | `robustCut`                           | `pd.cut` yields different results from cut in `r`    | Yes  |
| 14 | `removeOutliersQuantile`              | Yes                                                  | Yes  |
| 15 | `removeOutliersGLMfast`               | Error running the `r` script                         | No   |
| 16 | `compute_pValues_af`                  | Depends: `significanceTestPoisModel`                 | No   |
| 17 | `glm_mean_test`                       | Undefined: `quassipoisson`                           | No   |
