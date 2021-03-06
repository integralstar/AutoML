# AutoML (H2O) - ElectricalPowerOutput Prediction

### Data
AT|V|AP|RH|PE
---|---|---|---|---
 8.34|  40.77|  1010.84|  90.01|  480.48
23.64|  58.49|  1011.4|  74.2|   445.75
29.74|  56.9|   1007.15|  41.91|  438.76
19.0|  49.69|  1007.22|  76.79|  453.09
11.8|   40.66|  1017.13|  97.2|   464.43
13.97|  39.16|  1016.05|  84.6|   470.96
22.1|   71.29|  1008.2|   75.38|  442.35
14.47|  41.76|  1021.98|  78.41|  464
31.25|  69.51|  1010.25|  36.83|  428.77
 6.77|  38.18|  1017.8|   81.13|  484.31

### Leader Board
model_id|mean_residual_deviance|rmse|mse|mae|rmsle
---|---|---|---|---|---
StackedEnsemble_AllModels_AutoML_20210530|                      12.2783|  3.50404|  12.2783|  2.52853|  0.00769444
StackedEnsemble_BestOfFamily_AutoML_20210530|                   12.4833|  3.53317|  12.4833|  2.55067|  0.00775842
GBM_4_AutoML_20210530_042639|                                          12.5764|  3.54632|  12.5764|  2.55143|  0.00778255
GBM_3_AutoML_20210530_042639|                                          12.6131|  3.55149|  12.6131|  2.57656|  0.00779743
GBM_2_AutoML_20210530_042639|                                          13.0402|  3.61113|  13.0402|  2.63517|  0.00792932
GBM_1_AutoML_20210530_042639|                                          13.0783|  3.6164|   13.0783|  2.64601|  0.00793588
XRT_1_AutoML_20210530_042639|                                          13.3514|  3.65395|  13.3514|  2.64997|  0.00802118
DRF_1_AutoML_20210530_042639|                                          13.8051|  3.71552|  13.8051|  2.69759|  0.00815762
GBM_5_AutoML_20210530_042639|                                          14.9096|  3.86129|  14.9096|  2.88974|  0.00847027
DeepLearning_1_AutoML_20210530|                                 19.0893|  4.36913|  19.0893|  3.37842|  0.00959052
