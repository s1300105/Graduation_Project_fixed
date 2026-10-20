# Chemistry grid summary

Protocol: docs/CHEM_PROTOCOL_ja.md. Test ROC-AUC, mean +/- sd over seeds; 'best' = epoch with the best validation ROC-AUC, 'final' = last epoch.

## ogb_molbace

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 3 | 0.855 +/- 0.002 | 0.855 +/- 0.002 | 0.866 +/- 0.001 | 0.719 +/- 0.004 |
| A2 graph only | 3 | 0.748 +/- 0.033 | 0.789 +/- 0.007 | 0.747 +/- 0.027 | 0.634 +/- 0.008 |
| A2e graph, no edges | 3 | 0.685 +/- 0.005 | 0.692 +/- 0.003 | 0.686 +/- 0.013 | 0.649 +/- 0.008 |
| A1 text only | 3 | 0.805 +/- 0.009 | 0.801 +/- 0.010 | 0.822 +/- 0.009 | 0.727 +/- 0.010 |
| A3 hybrid | 2 | 0.812 +/- 0.000 | 0.799 +/- 0.024 | 0.812 +/- 0.000 | 0.708 +/- 0.007 |
| A3e hybrid, no edges | 2 | 0.794 +/- 0.001 | 0.818 +/- 0.021 | 0.799 +/- 0.005 | 0.718 +/- 0.003 |
| A3s hybrid, shuffled edges | 2 | 0.793 +/- 0.004 | 0.755 +/- 0.052 | 0.812 +/- 0.007 | 0.701 +/- 0.015 |

## ogb_molbbbp

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 2 | 0.679 +/- 0.002 | 0.679 +/- 0.002 | 0.716 +/- 0.006 | 0.958 +/- 0.002 |
| A2 graph only | 2 | 0.664 +/- 0.006 | 0.664 +/- 0.001 | 0.702 +/- 0.005 | 0.927 +/- 0.003 |
| A2e graph, no edges | 2 | 0.633 +/- 0.002 | 0.638 +/- 0.002 | 0.643 +/- 0.004 | 0.904 +/- 0.000 |
| A1 text only | 2 | 0.671 +/- 0.000 | 0.726 +/- 0.006 | 0.733 +/- 0.004 | 0.952 +/- 0.005 |
| A3 hybrid | 2 | 0.632 +/- 0.009 | 0.692 +/- 0.006 | 0.672 +/- 0.046 | 0.959 +/- 0.004 |
| A3e hybrid, no edges | 2 | 0.642 +/- 0.004 | 0.687 +/- 0.021 | 0.678 +/- 0.014 | 0.955 +/- 0.006 |
| A3s hybrid, shuffled edges | 2 | 0.658 +/- 0.012 | 0.685 +/- 0.012 | 0.711 +/- 0.027 | 0.954 +/- 0.000 |

### ogb_molbace: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 2/3 | +0.0069 | [-0.0314, +0.0439] | 0.0377 | 0.652 | 0.696 | 0.932 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 2/3 | +0.0069 | [-0.0267, +0.0425] | 0.0346 | 0.666 | 0.668 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 2/3 | -0.0019 | [-0.0530, +0.0458] | 0.0494 | 0.483 | 0.966 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 2/3 | -0.0019 | [-0.0401, +0.0375] | 0.0388 | 0.466 | 0.932 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 2/2 | +0.0174 | [-0.0053, +0.0418] | 0.0235 | 0.935 | 0.130 | 0.408 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 2/2 | +0.0174 | [-0.0003, +0.0358] | 0.0181 | 0.972 | 0.056 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 2/2 | -0.0190 | [-0.0659, +0.0220] | 0.0440 | 0.164 | 0.328 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 2/2 | -0.0190 | [-0.0395, -0.0003] | 0.0196 | 0.024 | 0.048 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 2/2 | +0.0182 | [-0.0196, +0.0544] | 0.0370 | 0.842 | 0.316 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 2/2 | +0.0182 | [-0.0131, +0.0495] | 0.0313 | 0.876 | 0.248 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 2/2 | +0.0446 | [-0.0397, +0.1249] | 0.0823 | 0.868 | 0.264 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 2/2 | +0.0446 | [-0.0032, +0.0924] | 0.0478 | 0.964 | 0.072 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 3/3 | +0.0629 | [+0.0177, +0.1306] | 0.0565 | 0.996 | 0.008 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 3/3 | +0.0629 | [+0.0250, +0.1031] | 0.0391 | 1.000 | 0.000 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 3/3 | +0.0969 | [+0.0203, +0.1735] | 0.0766 | 0.994 | 0.012 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 3/3 | +0.0969 | [+0.0241, +0.1705] | 0.0732 | 0.994 | 0.012 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 2/3 | -0.0437 | [-0.0978, +0.0087] | 0.0532 | 0.056 | 0.112 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 2/3 | -0.0437 | [-0.0980, +0.0084] | 0.0532 | 0.059 | 0.118 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 2/3 | -0.0560 | [-0.1178, +0.0005] | 0.0592 | 0.026 | 0.052 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 2/3 | -0.0560 | [-0.1091, -0.0047] | 0.0522 | 0.015 | 0.030 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 3/3 | -0.0506 | [-0.1021, +0.0003] | 0.0512 | 0.026 | 0.052 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 3/3 | -0.0506 | [-0.1043, -0.0008] | 0.0518 | 0.024 | 0.048 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 3/3 | -0.0542 | [-0.1074, -0.0022] | 0.0526 | 0.021 | 0.042 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 3/3 | -0.0542 | [-0.1079, -0.0039] | 0.0520 | 0.013 | 0.026 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 2/3 | +0.0634 | [-0.0098, +0.1415] | 0.0756 | 0.957 | 0.086 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 2/3 | +0.0634 | [-0.0028, +0.1271] | 0.0649 | 0.973 | 0.054 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 2/3 | +0.0100 | [-0.0626, +0.0797] | 0.0711 | 0.605 | 0.790 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 2/3 | +0.0100 | [-0.0563, +0.0776] | 0.0670 | 0.597 | 0.806 | - | - |

### ogb_molbbbp: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 2/2 | -0.0389 | [-0.0854, +0.0074] | 0.0464 | 0.051 | 0.102 | 0.408 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 2/2 | -0.0389 | [-0.0825, +0.0027] | 0.0426 | 0.033 | 0.066 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 2/2 | -0.0340 | [-0.0831, +0.0115] | 0.0473 | 0.086 | 0.172 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 2/2 | -0.0340 | [-0.0755, +0.0097] | 0.0426 | 0.062 | 0.124 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 2/2 | -0.0101 | [-0.0406, +0.0274] | 0.0340 | 0.233 | 0.466 | 0.932 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 2/2 | -0.0101 | [-0.0340, +0.0154] | 0.0247 | 0.204 | 0.408 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 2/2 | +0.0049 | [-0.0355, +0.0485] | 0.0420 | 0.590 | 0.820 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 2/2 | +0.0049 | [-0.0217, +0.0301] | 0.0259 | 0.650 | 0.700 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 2/2 | -0.0255 | [-0.0685, +0.0179] | 0.0432 | 0.106 | 0.212 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 2/2 | -0.0255 | [-0.0579, +0.0069] | 0.0324 | 0.058 | 0.116 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 2/2 | +0.0071 | [-0.0402, +0.0531] | 0.0466 | 0.619 | 0.762 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 2/2 | +0.0071 | [-0.0306, +0.0437] | 0.0372 | 0.648 | 0.704 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 2/2 | +0.0314 | [-0.0134, +0.0774] | 0.0454 | 0.910 | 0.180 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 2/2 | +0.0314 | [-0.0151, +0.0765] | 0.0458 | 0.910 | 0.180 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 2/2 | +0.0255 | [-0.0211, +0.0718] | 0.0465 | 0.853 | 0.294 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 2/2 | +0.0255 | [-0.0225, +0.0717] | 0.0471 | 0.851 | 0.298 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 2/2 | -0.0468 | [-0.1084, +0.0162] | 0.0623 | 0.067 | 0.134 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 2/2 | -0.0468 | [-0.1069, +0.0117] | 0.0593 | 0.063 | 0.126 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 2/2 | +0.0128 | [-0.0427, +0.0651] | 0.0539 | 0.673 | 0.654 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 2/2 | +0.0128 | [-0.0407, +0.0643] | 0.0525 | 0.683 | 0.634 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 2/2 | -0.0078 | [-0.0616, +0.0482] | 0.0549 | 0.381 | 0.762 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 2/2 | -0.0078 | [-0.0626, +0.0465] | 0.0546 | 0.389 | 0.778 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 2/2 | +0.0468 | [+0.0002, +0.0927] | 0.0462 | 0.976 | 0.048 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 2/2 | +0.0468 | [+0.0011, +0.0896] | 0.0442 | 0.977 | 0.046 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 2/2 | -0.0319 | [-0.0892, +0.0237] | 0.0565 | 0.133 | 0.266 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 2/2 | -0.0319 | [-0.0858, +0.0177] | 0.0518 | 0.113 | 0.226 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 2/2 | +0.0284 | [-0.0336, +0.0856] | 0.0596 | 0.824 | 0.352 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 2/2 | +0.0284 | [-0.0280, +0.0859] | 0.0570 | 0.833 | 0.334 | - | - |

