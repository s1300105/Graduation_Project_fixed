# Chemistry grid summary

Protocol: docs/CHEM_PROTOCOL_ja.md. Test ROC-AUC, mean +/- sd over seeds; 'best' = epoch with the best validation ROC-AUC, 'final' = last epoch.

## ogb_molbace

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 5 | 0.853 +/- 0.004 | 0.853 +/- 0.004 | 0.863 +/- 0.004 | 0.719 +/- 0.003 |
| A2 graph only | 5 | 0.745 +/- 0.027 | 0.787 +/- 0.007 | 0.749 +/- 0.027 | 0.641 +/- 0.011 |
| A2e graph, no edges | 5 | 0.688 +/- 0.008 | 0.691 +/- 0.004 | 0.693 +/- 0.015 | 0.655 +/- 0.010 |
| A1 text only | 5 | 0.805 +/- 0.011 | 0.801 +/- 0.010 | 0.821 +/- 0.011 | 0.731 +/- 0.009 |
| A3 hybrid | 5 | 0.811 +/- 0.018 | 0.803 +/- 0.020 | 0.819 +/- 0.008 | 0.705 +/- 0.006 |
| A3e hybrid, no edges | 5 | 0.802 +/- 0.013 | 0.816 +/- 0.019 | 0.808 +/- 0.009 | 0.709 +/- 0.009 |
| A3s hybrid, shuffled edges | 5 | 0.799 +/- 0.019 | 0.772 +/- 0.033 | 0.809 +/- 0.007 | 0.702 +/- 0.017 |

## ogb_molbbbp

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 5 | 0.677 +/- 0.004 | 0.677 +/- 0.004 | 0.715 +/- 0.004 | 0.959 +/- 0.001 |
| A2 graph only | 5 | 0.644 +/- 0.030 | 0.666 +/- 0.005 | 0.681 +/- 0.035 | 0.927 +/- 0.003 |
| A2e graph, no edges | 5 | 0.630 +/- 0.003 | 0.635 +/- 0.003 | 0.644 +/- 0.005 | 0.908 +/- 0.003 |
| A1 text only | 5 | 0.687 +/- 0.019 | 0.727 +/- 0.004 | 0.742 +/- 0.012 | 0.951 +/- 0.003 |
| A3 hybrid | 5 | 0.637 +/- 0.015 | 0.675 +/- 0.018 | 0.678 +/- 0.036 | 0.958 +/- 0.004 |
| A3e hybrid, no edges | 5 | 0.654 +/- 0.017 | 0.679 +/- 0.014 | 0.694 +/- 0.032 | 0.954 +/- 0.005 |
| A3s hybrid, shuffled edges | 5 | 0.649 +/- 0.015 | 0.672 +/- 0.018 | 0.688 +/- 0.035 | 0.953 +/- 0.007 |

### ogb_molbace: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 5/5 | +0.0066 | [-0.0300, +0.0433] | 0.0367 | 0.630 | 0.740 | 0.808 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 5/5 | +0.0066 | [-0.0252, +0.0393] | 0.0323 | 0.674 | 0.652 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 5/5 | +0.0023 | [-0.0373, +0.0430] | 0.0401 | 0.539 | 0.922 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 5/5 | +0.0023 | [-0.0314, +0.0377] | 0.0345 | 0.557 | 0.886 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 5/5 | +0.0098 | [-0.0159, +0.0347] | 0.0253 | 0.798 | 0.404 | 0.808 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 5/5 | +0.0098 | [-0.0008, +0.0212] | 0.0110 | 0.960 | 0.080 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 5/5 | -0.0129 | [-0.0442, +0.0169] | 0.0306 | 0.176 | 0.352 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 5/5 | -0.0129 | [-0.0274, +0.0012] | 0.0143 | 0.034 | 0.068 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 5/5 | +0.0125 | [-0.0241, +0.0419] | 0.0330 | 0.793 | 0.414 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 5/5 | +0.0125 | [-0.0070, +0.0317] | 0.0193 | 0.892 | 0.216 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 5/5 | +0.0308 | [-0.0174, +0.0850] | 0.0512 | 0.882 | 0.236 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 5/5 | +0.0308 | [-0.0029, +0.0646] | 0.0338 | 0.965 | 0.070 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 5/5 | +0.0571 | [+0.0193, +0.1040] | 0.0423 | 0.999 | 0.002 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 5/5 | +0.0571 | [+0.0244, +0.0905] | 0.0330 | 1.000 | 0.000 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 5/5 | +0.0961 | [+0.0209, +0.1686] | 0.0738 | 0.995 | 0.010 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 5/5 | +0.0961 | [+0.0247, +0.1704] | 0.0728 | 0.994 | 0.012 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 5/5 | -0.0417 | [-0.0939, +0.0111] | 0.0525 | 0.061 | 0.122 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 5/5 | -0.0417 | [-0.0928, +0.0077] | 0.0502 | 0.052 | 0.104 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 5/5 | -0.0499 | [-0.1069, +0.0007] | 0.0538 | 0.026 | 0.052 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 5/5 | -0.0499 | [-0.1015, -0.0028] | 0.0494 | 0.021 | 0.042 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 5/5 | -0.0483 | [-0.1022, +0.0035] | 0.0529 | 0.034 | 0.068 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 5/5 | -0.0483 | [-0.1018, +0.0010] | 0.0514 | 0.029 | 0.058 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 5/5 | -0.0522 | [-0.1076, -0.0006] | 0.0535 | 0.024 | 0.048 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 5/5 | -0.0522 | [-0.1069, -0.0020] | 0.0524 | 0.018 | 0.036 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 5/5 | +0.0659 | [-0.0094, +0.1425] | 0.0759 | 0.955 | 0.090 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 5/5 | +0.0659 | [-0.0019, +0.1321] | 0.0670 | 0.972 | 0.056 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 5/5 | +0.0159 | [-0.0506, +0.0805] | 0.0655 | 0.668 | 0.664 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 5/5 | +0.0159 | [-0.0459, +0.0813] | 0.0636 | 0.680 | 0.640 | - | - |

### ogb_molbbbp: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 5/5 | -0.0492 | [-0.0941, -0.0067] | 0.0437 | 0.013 | 0.026 | 0.104 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 5/5 | -0.0492 | [-0.0866, -0.0132] | 0.0367 | 0.001 | 0.002 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 5/5 | -0.0524 | [-0.1011, -0.0056] | 0.0478 | 0.013 | 0.026 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 5/5 | -0.0524 | [-0.0955, -0.0064] | 0.0445 | 0.012 | 0.024 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 5/5 | -0.0166 | [-0.0471, +0.0126] | 0.0299 | 0.134 | 0.268 | 0.804 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 5/5 | -0.0166 | [-0.0356, +0.0013] | 0.0185 | 0.033 | 0.066 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 5/5 | -0.0038 | [-0.0386, +0.0300] | 0.0343 | 0.409 | 0.818 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 5/5 | -0.0038 | [-0.0236, +0.0138] | 0.0187 | 0.350 | 0.700 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 5/5 | -0.0112 | [-0.0403, +0.0183] | 0.0293 | 0.218 | 0.436 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 5/5 | -0.0112 | [-0.0298, +0.0068] | 0.0183 | 0.107 | 0.214 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 5/5 | +0.0030 | [-0.0364, +0.0434] | 0.0399 | 0.575 | 0.850 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 5/5 | +0.0030 | [-0.0264, +0.0326] | 0.0295 | 0.584 | 0.832 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 5/5 | +0.0136 | [-0.0308, +0.0616] | 0.0462 | 0.722 | 0.556 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 5/5 | +0.0136 | [-0.0277, +0.0535] | 0.0406 | 0.741 | 0.518 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 5/5 | +0.0307 | [-0.0164, +0.0772] | 0.0468 | 0.903 | 0.194 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 5/5 | +0.0307 | [-0.0151, +0.0756] | 0.0454 | 0.906 | 0.188 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 5/5 | -0.0393 | [-0.0987, +0.0188] | 0.0587 | 0.084 | 0.168 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 5/5 | -0.0393 | [-0.0960, +0.0130] | 0.0545 | 0.085 | 0.170 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 5/5 | -0.0018 | [-0.0579, +0.0534] | 0.0557 | 0.474 | 0.948 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 5/5 | -0.0018 | [-0.0556, +0.0510] | 0.0533 | 0.477 | 0.954 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 5/5 | +0.0099 | [-0.0438, +0.0622] | 0.0530 | 0.649 | 0.702 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 5/5 | +0.0099 | [-0.0406, +0.0573] | 0.0489 | 0.664 | 0.672 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 5/5 | +0.0506 | [+0.0053, +0.0951] | 0.0449 | 0.984 | 0.032 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 5/5 | +0.0506 | [+0.0051, +0.0938] | 0.0444 | 0.986 | 0.028 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 5/5 | -0.0064 | [-0.0583, +0.0448] | 0.0515 | 0.411 | 0.822 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 5/5 | -0.0064 | [-0.0504, +0.0376] | 0.0440 | 0.381 | 0.762 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 5/5 | +0.0092 | [-0.0494, +0.0674] | 0.0584 | 0.619 | 0.762 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 5/5 | +0.0092 | [-0.0465, +0.0658] | 0.0561 | 0.621 | 0.758 | - | - |

