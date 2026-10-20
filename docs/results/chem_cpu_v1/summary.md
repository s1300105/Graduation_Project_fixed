# Chemistry grid summary

Protocol: docs/CHEM_PROTOCOL_ja.md. Test ROC-AUC, mean +/- sd over seeds; 'best' = epoch with the best validation ROC-AUC, 'final' = last epoch.

## ogb_molbace

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 4 | 0.853 +/- 0.004 | 0.853 +/- 0.004 | 0.863 +/- 0.004 | 0.719 +/- 0.004 |
| A2 graph only | 4 | 0.678 +/- 0.018 | 0.678 +/- 0.018 | 0.656 +/- 0.019 | 0.568 +/- 0.017 |
| A2e graph, no edges | 4 | 0.626 +/- 0.018 | 0.693 +/- 0.010 | 0.627 +/- 0.015 | 0.582 +/- 0.014 |
| A1 text only | 4 | 0.748 +/- 0.009 | 0.791 +/- 0.002 | 0.753 +/- 0.010 | 0.750 +/- 0.003 |
| A3 hybrid | 4 | 0.772 +/- 0.013 | 0.788 +/- 0.014 | 0.792 +/- 0.010 | 0.771 +/- 0.008 |
| A3e hybrid, no edges | 4 | 0.768 +/- 0.008 | 0.785 +/- 0.016 | 0.789 +/- 0.008 | 0.768 +/- 0.009 |
| A3s hybrid, shuffled edges | 4 | 0.769 +/- 0.011 | 0.782 +/- 0.017 | 0.791 +/- 0.010 | 0.769 +/- 0.007 |

## ogb_molbbbp

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 3 | 0.679 +/- 0.001 | 0.679 +/- 0.001 | 0.716 +/- 0.004 | 0.959 +/- 0.002 |
| A2 graph only | 3 | 0.599 +/- 0.006 | 0.595 +/- 0.010 | 0.598 +/- 0.014 | 0.925 +/- 0.002 |
| A2e graph, no edges | 3 | 0.630 +/- 0.003 | 0.630 +/- 0.003 | 0.668 +/- 0.003 | 0.911 +/- 0.003 |
| A1 text only | 3 | 0.661 +/- 0.004 | 0.700 +/- 0.005 | 0.706 +/- 0.004 | 0.959 +/- 0.001 |
| A3 hybrid | 3 | 0.674 +/- 0.007 | 0.706 +/- 0.009 | 0.724 +/- 0.008 | 0.959 +/- 0.002 |
| A3e hybrid, no edges | 3 | 0.665 +/- 0.025 | 0.716 +/- 0.005 | 0.710 +/- 0.021 | 0.958 +/- 0.001 |
| A3s hybrid, shuffled edges | 3 | 0.673 +/- 0.011 | 0.719 +/- 0.015 | 0.726 +/- 0.001 | 0.961 +/- 0.002 |

### ogb_molbace: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 4/4 | +0.0242 | [-0.0020, +0.0523] | 0.0272 | 0.963 | 0.074 | 0.296 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 4/4 | +0.0242 | [+0.0013, +0.0473] | 0.0230 | 0.982 | 0.036 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 4/4 | -0.0033 | [-0.0267, +0.0192] | 0.0229 | 0.393 | 0.786 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 4/4 | -0.0033 | [-0.0202, +0.0138] | 0.0170 | 0.347 | 0.694 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 4/4 | +0.0043 | [-0.0142, +0.0225] | 0.0183 | 0.698 | 0.604 | 1.000 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 4/4 | +0.0043 | [-0.0023, +0.0112] | 0.0067 | 0.905 | 0.190 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 4/4 | +0.0027 | [-0.0180, +0.0261] | 0.0221 | 0.591 | 0.818 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 4/4 | +0.0027 | [-0.0027, +0.0086] | 0.0057 | 0.827 | 0.346 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 4/4 | +0.0033 | [-0.0152, +0.0221] | 0.0186 | 0.642 | 0.716 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 4/4 | +0.0033 | [-0.0024, +0.0099] | 0.0062 | 0.861 | 0.278 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 4/4 | +0.0057 | [-0.0153, +0.0292] | 0.0222 | 0.690 | 0.620 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 4/4 | +0.0057 | [-0.0012, +0.0144] | 0.0078 | 0.939 | 0.122 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 4/4 | +0.0523 | [-0.0181, +0.1169] | 0.0675 | 0.934 | 0.132 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 4/4 | +0.0523 | [-0.0116, +0.1089] | 0.0602 | 0.945 | 0.110 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 4/4 | -0.0151 | [-0.0964, +0.0613] | 0.0789 | 0.337 | 0.674 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 4/4 | -0.0151 | [-0.0940, +0.0619] | 0.0779 | 0.324 | 0.648 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 4/4 | -0.0809 | [-0.1417, -0.0256] | 0.0580 | 0.003 | 0.006 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 4/4 | -0.0809 | [-0.1427, -0.0280] | 0.0573 | 0.002 | 0.004 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 4/4 | -0.0659 | [-0.1227, -0.0130] | 0.0549 | 0.006 | 0.012 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 4/4 | -0.0659 | [-0.1214, -0.0158] | 0.0528 | 0.005 | 0.010 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 4/4 | -0.1051 | [-0.1716, -0.0462] | 0.0627 | 0.000 | 0.000 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 4/4 | -0.1051 | [-0.1710, -0.0463] | 0.0624 | 0.000 | 0.000 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 4/4 | -0.0626 | [-0.1147, -0.0131] | 0.0508 | 0.005 | 0.010 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 4/4 | -0.0626 | [-0.1156, -0.0138] | 0.0509 | 0.006 | 0.012 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 4/4 | +0.0945 | [+0.0177, +0.1838] | 0.0830 | 0.989 | 0.022 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 4/4 | +0.0945 | [+0.0205, +0.1836] | 0.0816 | 0.993 | 0.014 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 4/4 | +0.1096 | [+0.0310, +0.1987] | 0.0838 | 0.995 | 0.010 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 4/4 | +0.1096 | [+0.0314, +0.1975] | 0.0831 | 0.997 | 0.006 | - | - |

### ogb_molbbbp: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 3/3 | +0.0127 | [-0.0205, +0.0439] | 0.0322 | 0.783 | 0.434 | 1.000 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 3/3 | +0.0127 | [-0.0151, +0.0415] | 0.0283 | 0.811 | 0.378 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 3/3 | +0.0063 | [-0.0435, +0.0538] | 0.0486 | 0.573 | 0.854 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 3/3 | +0.0063 | [-0.0394, +0.0514] | 0.0454 | 0.581 | 0.838 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 3/3 | +0.0094 | [-0.0231, +0.0434] | 0.0333 | 0.713 | 0.574 | 1.000 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 3/3 | +0.0094 | [-0.0034, +0.0220] | 0.0127 | 0.914 | 0.172 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 3/3 | -0.0102 | [-0.0313, +0.0072] | 0.0192 | 0.121 | 0.242 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 3/3 | -0.0102 | [-0.0219, +0.0005] | 0.0112 | 0.031 | 0.062 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 3/3 | +0.0008 | [-0.0214, +0.0223] | 0.0219 | 0.524 | 0.952 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 3/3 | +0.0008 | [-0.0073, +0.0093] | 0.0083 | 0.580 | 0.840 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 3/3 | -0.0131 | [-0.0380, +0.0120] | 0.0250 | 0.133 | 0.266 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 3/3 | -0.0131 | [-0.0255, -0.0022] | 0.0116 | 0.008 | 0.016 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 3/3 | -0.0309 | [-0.0777, +0.0154] | 0.0466 | 0.091 | 0.182 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 3/3 | -0.0309 | [-0.0777, +0.0123] | 0.0450 | 0.086 | 0.172 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 3/3 | -0.0349 | [-0.0832, +0.0131] | 0.0481 | 0.074 | 0.148 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 3/3 | -0.0349 | [-0.0830, +0.0105] | 0.0467 | 0.060 | 0.120 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 3/3 | -0.0048 | [-0.0631, +0.0565] | 0.0598 | 0.425 | 0.850 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 3/3 | -0.0048 | [-0.0641, +0.0525] | 0.0583 | 0.435 | 0.870 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 3/3 | +0.0271 | [-0.0339, +0.0849] | 0.0594 | 0.809 | 0.382 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 3/3 | +0.0271 | [-0.0316, +0.0855] | 0.0586 | 0.814 | 0.372 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 3/3 | -0.0175 | [-0.0734, +0.0380] | 0.0557 | 0.264 | 0.528 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 3/3 | -0.0175 | [-0.0729, +0.0355] | 0.0542 | 0.257 | 0.514 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 3/3 | +0.0208 | [-0.0291, +0.0726] | 0.0508 | 0.803 | 0.394 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 3/3 | +0.0208 | [-0.0268, +0.0671] | 0.0470 | 0.806 | 0.388 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 3/3 | +0.0753 | [+0.0248, +0.1292] | 0.0522 | 0.999 | 0.002 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 3/3 | +0.0753 | [+0.0279, +0.1272] | 0.0496 | 1.000 | 0.000 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 3/3 | +0.1113 | [+0.0471, +0.1782] | 0.0655 | 1.000 | 0.000 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 3/3 | +0.1113 | [+0.0508, +0.1745] | 0.0618 | 1.000 | 0.000 | - | - |

