# Chemistry grid summary

Protocol: docs/CHEM_PROTOCOL_ja.md. Test ROC-AUC, mean +/- sd over seeds; 'best' = epoch with the best validation ROC-AUC, 'final' = last epoch.

## ogb_molbace

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 2 | 0.855 +/- 0.003 | 0.855 +/- 0.003 | 0.865 +/- 0.001 | 0.721 +/- 0.003 |
| A2 graph only | 2 | 0.729 +/- 0.008 | 0.793 +/- 0.001 | 0.732 +/- 0.001 | 0.636 +/- 0.010 |
| A2e graph, no edges | 2 | 0.688 +/- 0.001 | 0.690 +/- 0.001 | 0.685 +/- 0.017 | 0.650 +/- 0.012 |
| A1 text only | 1 | 0.812 +/- 0.000 | 0.809 +/- 0.000 | 0.824 +/- 0.000 | 0.729 +/- 0.000 |
| A3 hybrid | 1 | 0.812 +/- 0.000 | 0.782 +/- 0.000 | 0.812 +/- 0.000 | 0.703 +/- 0.000 |
| A3e hybrid, no edges | 1 | 0.795 +/- 0.000 | 0.803 +/- 0.000 | 0.795 +/- 0.000 | 0.716 +/- 0.000 |
| A3s hybrid, shuffled edges | 1 | 0.797 +/- 0.000 | 0.718 +/- 0.000 | 0.817 +/- 0.000 | 0.712 +/- 0.000 |

## ogb_molbbbp

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 1 | 0.681 +/- 0.000 | 0.681 +/- 0.000 | 0.712 +/- 0.000 | 0.957 +/- 0.000 |
| A2 graph only | 1 | 0.668 +/- 0.000 | 0.663 +/- 0.000 | 0.706 +/- 0.000 | 0.924 +/- 0.000 |
| A2e graph, no edges | 1 | 0.634 +/- 0.000 | 0.639 +/- 0.000 | 0.646 +/- 0.000 | 0.904 +/- 0.000 |
| A1 text only | 1 | 0.671 +/- 0.000 | 0.730 +/- 0.000 | 0.736 +/- 0.000 | 0.948 +/- 0.000 |
| A3 hybrid | 1 | 0.626 +/- 0.000 | 0.687 +/- 0.000 | 0.639 +/- 0.000 | 0.962 +/- 0.000 |
| A3e hybrid, no edges | 1 | 0.639 +/- 0.000 | 0.702 +/- 0.000 | 0.668 +/- 0.000 | 0.959 +/- 0.000 |
| A3s hybrid, shuffled edges | 1 | 0.650 +/- 0.000 | 0.694 +/- 0.000 | 0.692 +/- 0.000 | 0.954 +/- 0.000 |

### ogb_molbace: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 1/1 | +0.0000 | [-0.0415, +0.0411] | 0.0413 | 0.497 | 0.994 | 0.994 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 1/1 | +0.0000 | [-0.0415, +0.0411] | 0.0413 | 0.497 | 0.994 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 1/1 | -0.0275 | [-0.0718, +0.0183] | 0.0450 | 0.131 | 0.262 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 1/1 | -0.0275 | [-0.0718, +0.0183] | 0.0450 | 0.131 | 0.262 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 1/1 | +0.0170 | [-0.0051, +0.0394] | 0.0222 | 0.928 | 0.144 | 0.432 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 1/1 | +0.0170 | [-0.0051, +0.0394] | 0.0222 | 0.928 | 0.144 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 1/1 | -0.0212 | [-0.0505, +0.0056] | 0.0281 | 0.056 | 0.112 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 1/1 | -0.0212 | [-0.0505, +0.0056] | 0.0281 | 0.056 | 0.112 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 1/1 | +0.0153 | [-0.0326, +0.0624] | 0.0475 | 0.727 | 0.546 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 1/1 | +0.0153 | [-0.0326, +0.0624] | 0.0475 | 0.727 | 0.546 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 1/1 | +0.0640 | [-0.0000, +0.1287] | 0.0644 | 0.975 | 0.050 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 1/1 | +0.0640 | [-0.0000, +0.1287] | 0.0644 | 0.975 | 0.050 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 2/2 | +0.0410 | [+0.0071, +0.0759] | 0.0344 | 0.992 | 0.016 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 2/2 | +0.0410 | [+0.0116, +0.0691] | 0.0287 | 0.997 | 0.006 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 2/2 | +0.1029 | [+0.0270, +0.1810] | 0.0770 | 0.998 | 0.004 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 2/2 | +0.1029 | [+0.0287, +0.1765] | 0.0739 | 0.996 | 0.008 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 1/2 | -0.0431 | [-0.1039, +0.0156] | 0.0598 | 0.067 | 0.134 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 1/2 | -0.0431 | [-0.1027, +0.0131] | 0.0579 | 0.074 | 0.148 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 1/2 | -0.0730 | [-0.1335, -0.0169] | 0.0583 | 0.006 | 0.012 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 1/2 | -0.0730 | [-0.1312, -0.0158] | 0.0577 | 0.004 | 0.008 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 1/2 | -0.0431 | [-0.0999, +0.0096] | 0.0548 | 0.053 | 0.106 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 1/2 | -0.0431 | [-0.0993, +0.0085] | 0.0539 | 0.048 | 0.096 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 1/2 | -0.0455 | [-0.1015, +0.0064] | 0.0540 | 0.043 | 0.086 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 1/2 | -0.0455 | [-0.1003, +0.0055] | 0.0529 | 0.043 | 0.086 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 1/2 | +0.0826 | [+0.0021, +0.1634] | 0.0806 | 0.978 | 0.044 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 1/2 | +0.0826 | [+0.0044, +0.1624] | 0.0790 | 0.978 | 0.044 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 1/2 | -0.0114 | [-0.0845, +0.0650] | 0.0748 | 0.368 | 0.736 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 1/2 | -0.0114 | [-0.0834, +0.0620] | 0.0727 | 0.370 | 0.740 | - | - |

### ogb_molbbbp: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 1/1 | -0.0451 | [-0.0900, -0.0044] | 0.0428 | 0.013 | 0.026 | 0.104 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 1/1 | -0.0451 | [-0.0900, -0.0044] | 0.0428 | 0.013 | 0.026 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 1/1 | -0.0430 | [-0.0952, +0.0083] | 0.0518 | 0.051 | 0.102 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 1/1 | -0.0430 | [-0.0952, +0.0083] | 0.0518 | 0.051 | 0.102 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 1/1 | -0.0133 | [-0.0419, +0.0116] | 0.0267 | 0.164 | 0.328 | 0.656 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 1/1 | -0.0133 | [-0.0419, +0.0116] | 0.0267 | 0.164 | 0.328 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 1/1 | -0.0145 | [-0.0534, +0.0214] | 0.0374 | 0.208 | 0.416 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 1/1 | -0.0145 | [-0.0534, +0.0214] | 0.0374 | 0.208 | 0.416 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 1/1 | -0.0236 | [-0.0652, +0.0140] | 0.0396 | 0.113 | 0.226 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 1/1 | -0.0236 | [-0.0652, +0.0140] | 0.0396 | 0.113 | 0.226 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 1/1 | -0.0061 | [-0.0552, +0.0426] | 0.0489 | 0.401 | 0.802 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 1/1 | -0.0061 | [-0.0552, +0.0426] | 0.0489 | 0.401 | 0.802 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 1/1 | +0.0340 | [-0.0143, +0.0816] | 0.0480 | 0.916 | 0.168 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 1/1 | +0.0340 | [-0.0143, +0.0816] | 0.0480 | 0.916 | 0.168 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 1/1 | +0.0231 | [-0.0302, +0.0727] | 0.0515 | 0.805 | 0.390 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 1/1 | +0.0231 | [-0.0302, +0.0727] | 0.0515 | 0.805 | 0.390 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 1/1 | -0.0545 | [-0.1152, +0.0036] | 0.0594 | 0.033 | 0.066 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 1/1 | -0.0545 | [-0.1152, +0.0036] | 0.0594 | 0.033 | 0.066 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 1/1 | +0.0069 | [-0.0566, +0.0670] | 0.0618 | 0.591 | 0.818 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 1/1 | +0.0069 | [-0.0566, +0.0670] | 0.0618 | 0.591 | 0.818 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 1/1 | -0.0094 | [-0.0692, +0.0490] | 0.0591 | 0.372 | 0.744 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 1/1 | -0.0094 | [-0.0692, +0.0490] | 0.0591 | 0.372 | 0.744 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 1/1 | +0.0499 | [+0.0024, +0.0953] | 0.0464 | 0.982 | 0.036 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 1/1 | +0.0499 | [+0.0024, +0.0953] | 0.0464 | 0.982 | 0.036 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 1/1 | -0.0421 | [-0.0983, +0.0106] | 0.0544 | 0.058 | 0.116 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 1/1 | -0.0421 | [-0.0983, +0.0106] | 0.0544 | 0.058 | 0.116 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 1/1 | +0.0249 | [-0.0433, +0.0890] | 0.0662 | 0.765 | 0.470 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 1/1 | +0.0249 | [-0.0433, +0.0890] | 0.0662 | 0.765 | 0.470 | - | - |

