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
| A2e graph, no edges | 4 | 0.631 +/- 0.003 | 0.636 +/- 0.003 | 0.644 +/- 0.005 | 0.907 +/- 0.003 |
| A1 text only | 4 | 0.689 +/- 0.021 | 0.727 +/- 0.005 | 0.743 +/- 0.013 | 0.950 +/- 0.003 |
| A3 hybrid | 4 | 0.642 +/- 0.013 | 0.677 +/- 0.020 | 0.687 +/- 0.034 | 0.957 +/- 0.005 |
| A3e hybrid, no edges | 4 | 0.657 +/- 0.019 | 0.679 +/- 0.016 | 0.701 +/- 0.033 | 0.954 +/- 0.005 |
| A3s hybrid, shuffled edges | 4 | 0.655 +/- 0.008 | 0.675 +/- 0.019 | 0.702 +/- 0.019 | 0.954 +/- 0.007 |

### ogb_molbace: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 5/5 | +0.0066 | [-0.0300, +0.0433] | 0.0367 | 0.630 | 0.740 | 1.000 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 5/5 | +0.0066 | [-0.0252, +0.0393] | 0.0323 | 0.674 | 0.652 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 5/5 | +0.0023 | [-0.0373, +0.0430] | 0.0401 | 0.539 | 0.922 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 5/5 | +0.0023 | [-0.0314, +0.0377] | 0.0345 | 0.557 | 0.886 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 5/5 | +0.0098 | [-0.0159, +0.0347] | 0.0253 | 0.798 | 0.404 | 1.000 | no difference shown |
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
| primary | A3 hybrid - A1 text only | best | two_level | 4/4 | -0.0476 | [-0.0935, -0.0041] | 0.0447 | 0.012 | 0.024 | 0.096 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 4/4 | -0.0476 | [-0.0865, -0.0101] | 0.0382 | 0.006 | 0.012 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 4/4 | -0.0499 | [-0.1003, -0.0022] | 0.0491 | 0.021 | 0.042 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 4/4 | -0.0499 | [-0.0908, -0.0049] | 0.0430 | 0.015 | 0.030 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 4/4 | -0.0148 | [-0.0478, +0.0159] | 0.0319 | 0.173 | 0.346 | 1.000 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 4/4 | -0.0148 | [-0.0373, +0.0063] | 0.0218 | 0.086 | 0.172 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 4/4 | -0.0018 | [-0.0342, +0.0321] | 0.0331 | 0.464 | 0.928 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 4/4 | -0.0018 | [-0.0217, +0.0179] | 0.0198 | 0.443 | 0.886 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 4/4 | -0.0129 | [-0.0426, +0.0144] | 0.0285 | 0.176 | 0.352 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 4/4 | -0.0129 | [-0.0330, +0.0067] | 0.0198 | 0.089 | 0.178 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 4/4 | +0.0020 | [-0.0414, +0.0451] | 0.0432 | 0.542 | 0.916 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 4/4 | +0.0020 | [-0.0284, +0.0320] | 0.0302 | 0.553 | 0.894 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 5/4 | +0.0125 | [-0.0334, +0.0617] | 0.0475 | 0.700 | 0.600 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 5/4 | +0.0125 | [-0.0288, +0.0515] | 0.0401 | 0.718 | 0.564 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 5/4 | +0.0296 | [-0.0138, +0.0765] | 0.0451 | 0.895 | 0.210 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 5/4 | +0.0296 | [-0.0155, +0.0742] | 0.0449 | 0.897 | 0.206 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 4/5 | -0.0349 | [-0.0948, +0.0214] | 0.0581 | 0.118 | 0.236 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 4/5 | -0.0349 | [-0.0923, +0.0178] | 0.0550 | 0.111 | 0.222 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 4/5 | +0.0006 | [-0.0553, +0.0559] | 0.0556 | 0.510 | 0.980 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 4/5 | +0.0006 | [-0.0525, +0.0529] | 0.0527 | 0.508 | 0.984 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 4/5 | +0.0128 | [-0.0441, +0.0629] | 0.0535 | 0.684 | 0.632 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 4/5 | +0.0128 | [-0.0377, +0.0601] | 0.0489 | 0.706 | 0.588 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 4/5 | +0.0505 | [+0.0054, +0.0956] | 0.0451 | 0.984 | 0.032 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 4/5 | +0.0505 | [+0.0041, +0.0948] | 0.0453 | 0.986 | 0.028 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 4/5 | -0.0020 | [-0.0532, +0.0509] | 0.0520 | 0.454 | 0.908 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 4/5 | -0.0020 | [-0.0478, +0.0426] | 0.0452 | 0.450 | 0.900 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 4/5 | +0.0116 | [-0.0478, +0.0663] | 0.0571 | 0.646 | 0.708 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 4/5 | +0.0116 | [-0.0418, +0.0675] | 0.0547 | 0.659 | 0.682 | - | - |

