# Chemistry grid summary

Protocol: docs/CHEM_PROTOCOL_ja.md. Test ROC-AUC, mean +/- sd over seeds; 'best' = epoch with the best validation ROC-AUC, 'final' = last epoch.

## ogb_molbace

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 4 | 0.853 +/- 0.004 | 0.853 +/- 0.004 | 0.863 +/- 0.004 | 0.719 +/- 0.004 |
| A2 graph only | 4 | 0.742 +/- 0.030 | 0.786 +/- 0.008 | 0.742 +/- 0.025 | 0.637 +/- 0.008 |
| A2e graph, no edges | 4 | 0.685 +/- 0.004 | 0.692 +/- 0.003 | 0.688 +/- 0.011 | 0.653 +/- 0.011 |
| A1 text only | 4 | 0.808 +/- 0.010 | 0.803 +/- 0.009 | 0.825 +/- 0.009 | 0.729 +/- 0.009 |
| A3 hybrid | 4 | 0.819 +/- 0.010 | 0.808 +/- 0.019 | 0.820 +/- 0.009 | 0.707 +/- 0.005 |
| A3e hybrid, no edges | 4 | 0.804 +/- 0.014 | 0.822 +/- 0.015 | 0.807 +/- 0.010 | 0.709 +/- 0.010 |
| A3s hybrid, shuffled edges | 3 | 0.800 +/- 0.012 | 0.768 +/- 0.044 | 0.808 +/- 0.008 | 0.700 +/- 0.011 |

## ogb_molbbbp

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 3 | 0.679 +/- 0.001 | 0.679 +/- 0.001 | 0.716 +/- 0.004 | 0.959 +/- 0.002 |
| A2 graph only | 3 | 0.650 +/- 0.025 | 0.664 +/- 0.002 | 0.686 +/- 0.028 | 0.926 +/- 0.003 |
| A2e graph, no edges | 3 | 0.633 +/- 0.001 | 0.636 +/- 0.003 | 0.645 +/- 0.005 | 0.906 +/- 0.003 |
| A1 text only | 3 | 0.682 +/- 0.019 | 0.726 +/- 0.005 | 0.737 +/- 0.007 | 0.951 +/- 0.004 |
| A3 hybrid | 3 | 0.636 +/- 0.009 | 0.678 +/- 0.025 | 0.677 +/- 0.034 | 0.956 +/- 0.006 |
| A3e hybrid, no edges | 3 | 0.649 +/- 0.011 | 0.680 +/- 0.019 | 0.687 +/- 0.018 | 0.953 +/- 0.005 |
| A3s hybrid, shuffled edges | 3 | 0.657 +/- 0.008 | 0.673 +/- 0.022 | 0.704 +/- 0.023 | 0.951 +/- 0.005 |

### ogb_molbace: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 4/4 | +0.0106 | [-0.0242, +0.0467] | 0.0354 | 0.726 | 0.548 | 0.804 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 4/4 | +0.0106 | [-0.0210, +0.0439] | 0.0324 | 0.740 | 0.520 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 4/4 | +0.0051 | [-0.0355, +0.0447] | 0.0401 | 0.602 | 0.796 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 4/4 | +0.0051 | [-0.0283, +0.0406] | 0.0344 | 0.623 | 0.754 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 4/4 | +0.0148 | [-0.0086, +0.0390] | 0.0238 | 0.887 | 0.226 | 0.678 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 4/4 | +0.0148 | [+0.0030, +0.0286] | 0.0128 | 0.993 | 0.014 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 4/4 | -0.0137 | [-0.0439, +0.0127] | 0.0283 | 0.172 | 0.344 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 4/4 | -0.0137 | [-0.0297, +0.0004] | 0.0151 | 0.029 | 0.058 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 4/3 | +0.0182 | [-0.0179, +0.0544] | 0.0362 | 0.848 | 0.304 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 4/3 | +0.0182 | [-0.0091, +0.0443] | 0.0267 | 0.907 | 0.186 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 4/3 | +0.0399 | [-0.0211, +0.1082] | 0.0646 | 0.896 | 0.208 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 4/3 | +0.0399 | [-0.0004, +0.0805] | 0.0404 | 0.973 | 0.054 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 4/4 | +0.0565 | [+0.0170, +0.1091] | 0.0461 | 0.999 | 0.002 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 4/4 | +0.0565 | [+0.0239, +0.0905] | 0.0333 | 1.000 | 0.000 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 4/4 | +0.0939 | [+0.0218, +0.1691] | 0.0736 | 0.997 | 0.006 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 4/4 | +0.0939 | [+0.0229, +0.1679] | 0.0725 | 0.994 | 0.012 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 4/4 | -0.0348 | [-0.0866, +0.0143] | 0.0504 | 0.084 | 0.168 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 4/4 | -0.0348 | [-0.0854, +0.0153] | 0.0503 | 0.091 | 0.182 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 4/4 | -0.0451 | [-0.0993, +0.0046] | 0.0519 | 0.038 | 0.076 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 4/4 | -0.0451 | [-0.0956, +0.0017] | 0.0486 | 0.030 | 0.060 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 4/4 | -0.0454 | [-0.0970, +0.0051] | 0.0510 | 0.038 | 0.076 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 4/4 | -0.0454 | [-0.0979, +0.0040] | 0.0510 | 0.035 | 0.070 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 4/4 | -0.0501 | [-0.1022, +0.0020] | 0.0521 | 0.030 | 0.060 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 4/4 | -0.0501 | [-0.1040, -0.0001] | 0.0519 | 0.025 | 0.050 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 4/4 | +0.0769 | [+0.0078, +0.1502] | 0.0712 | 0.984 | 0.032 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 4/4 | +0.0769 | [+0.0108, +0.1415] | 0.0653 | 0.991 | 0.018 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 4/4 | +0.0220 | [-0.0479, +0.0882] | 0.0680 | 0.728 | 0.544 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 4/4 | +0.0220 | [-0.0394, +0.0859] | 0.0626 | 0.756 | 0.488 | - | - |

### ogb_molbbbp: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 3/3 | -0.0460 | [-0.0943, -0.0011] | 0.0466 | 0.022 | 0.044 | 0.176 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 3/3 | -0.0460 | [-0.0857, -0.0064] | 0.0396 | 0.007 | 0.014 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 3/3 | -0.0480 | [-0.1000, +0.0040] | 0.0520 | 0.035 | 0.070 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 3/3 | -0.0480 | [-0.0906, -0.0001] | 0.0452 | 0.025 | 0.050 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 3/3 | -0.0122 | [-0.0445, +0.0189] | 0.0317 | 0.201 | 0.402 | 0.804 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 3/3 | -0.0122 | [-0.0344, +0.0102] | 0.0223 | 0.134 | 0.268 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 3/3 | -0.0024 | [-0.0431, +0.0372] | 0.0401 | 0.440 | 0.880 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 3/3 | -0.0024 | [-0.0248, +0.0179] | 0.0214 | 0.434 | 0.868 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 3/3 | -0.0204 | [-0.0521, +0.0098] | 0.0309 | 0.083 | 0.166 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 3/3 | -0.0204 | [-0.0429, +0.0003] | 0.0216 | 0.025 | 0.050 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 3/3 | +0.0046 | [-0.0461, +0.0538] | 0.0500 | 0.577 | 0.846 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 3/3 | +0.0046 | [-0.0283, +0.0373] | 0.0328 | 0.612 | 0.776 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 3/3 | +0.0173 | [-0.0303, +0.0675] | 0.0489 | 0.748 | 0.504 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 3/3 | +0.0173 | [-0.0245, +0.0582] | 0.0413 | 0.786 | 0.428 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 3/3 | +0.0278 | [-0.0191, +0.0727] | 0.0459 | 0.881 | 0.238 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 3/3 | +0.0278 | [-0.0189, +0.0726] | 0.0457 | 0.881 | 0.238 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 3/3 | -0.0427 | [-0.1022, +0.0162] | 0.0592 | 0.075 | 0.150 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 3/3 | -0.0427 | [-0.1028, +0.0120] | 0.0574 | 0.071 | 0.142 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 3/3 | -0.0012 | [-0.0626, +0.0564] | 0.0595 | 0.489 | 0.978 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 3/3 | -0.0012 | [-0.0540, +0.0507] | 0.0524 | 0.484 | 0.968 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 3/3 | +0.0033 | [-0.0518, +0.0556] | 0.0537 | 0.539 | 0.922 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 3/3 | +0.0033 | [-0.0492, +0.0522] | 0.0507 | 0.554 | 0.892 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 3/3 | +0.0468 | [-0.0017, +0.0934] | 0.0475 | 0.970 | 0.060 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 3/3 | +0.0468 | [+0.0004, +0.0907] | 0.0452 | 0.977 | 0.046 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 3/3 | -0.0136 | [-0.0728, +0.0414] | 0.0571 | 0.310 | 0.620 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 3/3 | -0.0136 | [-0.0617, +0.0327] | 0.0472 | 0.275 | 0.550 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 3/3 | +0.0135 | [-0.0481, +0.0722] | 0.0601 | 0.663 | 0.674 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 3/3 | +0.0135 | [-0.0426, +0.0695] | 0.0561 | 0.679 | 0.642 | - | - |

