# Chemistry grid summary

Protocol: docs/CHEM_PROTOCOL_ja.md. Test ROC-AUC, mean +/- sd over seeds; 'best' = epoch with the best validation ROC-AUC, 'final' = last epoch.

## ogb_molbace

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 5 | 0.853 +/- 0.004 | 0.853 +/- 0.004 | 0.863 +/- 0.004 | 0.719 +/- 0.003 |
| A2 graph only | 5 | 0.680 +/- 0.016 | 0.681 +/- 0.017 | 0.657 +/- 0.017 | 0.568 +/- 0.015 |
| A2e graph, no edges | 5 | 0.619 +/- 0.021 | 0.692 +/- 0.009 | 0.626 +/- 0.013 | 0.584 +/- 0.013 |
| A1 text only | 5 | 0.744 +/- 0.012 | 0.785 +/- 0.013 | 0.752 +/- 0.009 | 0.753 +/- 0.009 |
| A3 hybrid | 5 | 0.773 +/- 0.011 | 0.787 +/- 0.012 | 0.792 +/- 0.009 | 0.770 +/- 0.008 |
| A3e hybrid, no edges | 5 | 0.769 +/- 0.007 | 0.786 +/- 0.014 | 0.789 +/- 0.007 | 0.767 +/- 0.008 |
| A3s hybrid, shuffled edges | 5 | 0.770 +/- 0.009 | 0.783 +/- 0.015 | 0.792 +/- 0.009 | 0.768 +/- 0.006 |

## ogb_molbbbp

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 5 | 0.677 +/- 0.004 | 0.677 +/- 0.004 | 0.715 +/- 0.004 | 0.959 +/- 0.001 |
| A2 graph only | 5 | 0.596 +/- 0.008 | 0.593 +/- 0.009 | 0.601 +/- 0.014 | 0.926 +/- 0.003 |
| A2e graph, no edges | 5 | 0.629 +/- 0.004 | 0.630 +/- 0.004 | 0.668 +/- 0.006 | 0.911 +/- 0.002 |
| A1 text only | 5 | 0.667 +/- 0.010 | 0.701 +/- 0.005 | 0.709 +/- 0.007 | 0.958 +/- 0.002 |
| A3 hybrid | 5 | 0.663 +/- 0.022 | 0.710 +/- 0.009 | 0.706 +/- 0.038 | 0.958 +/- 0.003 |
| A3e hybrid, no edges | 5 | 0.658 +/- 0.026 | 0.717 +/- 0.004 | 0.697 +/- 0.035 | 0.957 +/- 0.003 |
| A3s hybrid, shuffled edges | 5 | 0.663 +/- 0.022 | 0.713 +/- 0.015 | 0.710 +/- 0.033 | 0.960 +/- 0.002 |

### ogb_molbace: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 5/5 | +0.0287 | [+0.0018, +0.0561] | 0.0271 | 0.984 | 0.032 | 0.128 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 5/5 | +0.0287 | [+0.0054, +0.0518] | 0.0232 | 0.994 | 0.012 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 5/5 | +0.0015 | [-0.0207, +0.0245] | 0.0226 | 0.541 | 0.918 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 5/5 | +0.0015 | [-0.0143, +0.0167] | 0.0155 | 0.563 | 0.874 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 5/5 | +0.0040 | [-0.0127, +0.0193] | 0.0160 | 0.728 | 0.544 | 1.000 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 5/5 | +0.0040 | [-0.0015, +0.0099] | 0.0057 | 0.930 | 0.140 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 5/5 | +0.0010 | [-0.0176, +0.0199] | 0.0188 | 0.566 | 0.868 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 5/5 | +0.0010 | [-0.0039, +0.0062] | 0.0050 | 0.658 | 0.684 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 5/5 | +0.0028 | [-0.0143, +0.0197] | 0.0170 | 0.649 | 0.702 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 5/5 | +0.0028 | [-0.0034, +0.0091] | 0.0063 | 0.806 | 0.388 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 5/5 | +0.0034 | [-0.0146, +0.0245] | 0.0196 | 0.652 | 0.696 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 5/5 | +0.0034 | [-0.0032, +0.0111] | 0.0072 | 0.822 | 0.356 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 5/5 | +0.0605 | [-0.0055, +0.1233] | 0.0644 | 0.965 | 0.070 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 5/5 | +0.0605 | [-0.0030, +0.1187] | 0.0609 | 0.970 | 0.060 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 5/5 | -0.0111 | [-0.0906, +0.0658] | 0.0782 | 0.371 | 0.742 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 5/5 | -0.0111 | [-0.0885, +0.0646] | 0.0766 | 0.365 | 0.730 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 5/5 | -0.0801 | [-0.1410, -0.0243] | 0.0583 | 0.003 | 0.006 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 5/5 | -0.0801 | [-0.1424, -0.0263] | 0.0580 | 0.003 | 0.006 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 5/5 | -0.0663 | [-0.1241, -0.0129] | 0.0556 | 0.005 | 0.010 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 5/5 | -0.0663 | [-0.1220, -0.0163] | 0.0529 | 0.005 | 0.010 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 5/5 | -0.1088 | [-0.1755, -0.0470] | 0.0643 | 0.000 | 0.000 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 5/5 | -0.1088 | [-0.1755, -0.0487] | 0.0634 | 0.000 | 0.000 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 5/5 | -0.0678 | [-0.1245, -0.0146] | 0.0549 | 0.004 | 0.008 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 5/5 | -0.0678 | [-0.1211, -0.0186] | 0.0513 | 0.004 | 0.008 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 5/5 | +0.0931 | [+0.0139, +0.1768] | 0.0815 | 0.993 | 0.014 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 5/5 | +0.0931 | [+0.0194, +0.1817] | 0.0811 | 0.993 | 0.014 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 5/5 | +0.1056 | [+0.0265, +0.1906] | 0.0820 | 0.997 | 0.006 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 5/5 | +0.1056 | [+0.0286, +0.1924] | 0.0819 | 0.996 | 0.008 | - | - |

### ogb_molbbbp: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 5/5 | -0.0040 | [-0.0411, +0.0290] | 0.0350 | 0.411 | 0.822 | 1.000 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 5/5 | -0.0040 | [-0.0320, +0.0258] | 0.0289 | 0.372 | 0.744 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 5/5 | +0.0092 | [-0.0359, +0.0535] | 0.0447 | 0.629 | 0.742 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 5/5 | +0.0092 | [-0.0328, +0.0521] | 0.0425 | 0.637 | 0.726 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 5/5 | +0.0052 | [-0.0287, +0.0386] | 0.0336 | 0.628 | 0.744 | 1.000 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 5/5 | +0.0052 | [-0.0040, +0.0145] | 0.0093 | 0.851 | 0.298 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 5/5 | -0.0072 | [-0.0246, +0.0077] | 0.0162 | 0.163 | 0.326 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 5/5 | -0.0072 | [-0.0157, +0.0007] | 0.0082 | 0.040 | 0.080 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 5/5 | -0.0005 | [-0.0310, +0.0288] | 0.0299 | 0.508 | 0.984 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 5/5 | -0.0005 | [-0.0059, +0.0053] | 0.0056 | 0.436 | 0.872 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 5/5 | -0.0030 | [-0.0230, +0.0167] | 0.0198 | 0.382 | 0.764 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 5/5 | -0.0030 | [-0.0106, +0.0047] | 0.0077 | 0.220 | 0.440 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 5/5 | -0.0335 | [-0.0805, +0.0124] | 0.0464 | 0.076 | 0.152 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 5/5 | -0.0335 | [-0.0823, +0.0110] | 0.0466 | 0.078 | 0.156 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 5/5 | -0.0361 | [-0.0837, +0.0107] | 0.0472 | 0.065 | 0.130 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 5/5 | -0.0361 | [-0.0850, +0.0095] | 0.0473 | 0.061 | 0.122 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 5/5 | -0.0139 | [-0.0727, +0.0476] | 0.0602 | 0.327 | 0.654 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 5/5 | -0.0139 | [-0.0704, +0.0429] | 0.0567 | 0.302 | 0.604 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 5/5 | +0.0333 | [-0.0237, +0.0912] | 0.0575 | 0.862 | 0.276 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 5/5 | +0.0333 | [-0.0236, +0.0901] | 0.0568 | 0.874 | 0.252 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 5/5 | -0.0098 | [-0.0645, +0.0444] | 0.0544 | 0.356 | 0.712 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 5/5 | -0.0098 | [-0.0621, +0.0419] | 0.0520 | 0.351 | 0.702 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 5/5 | +0.0241 | [-0.0228, +0.0718] | 0.0473 | 0.837 | 0.326 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 5/5 | +0.0241 | [-0.0224, +0.0705] | 0.0464 | 0.844 | 0.312 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 5/5 | +0.0669 | [+0.0144, +0.1233] | 0.0545 | 0.993 | 0.014 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 5/5 | +0.0669 | [+0.0186, +0.1188] | 0.0501 | 0.998 | 0.004 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 5/5 | +0.1165 | [+0.0541, +0.1774] | 0.0616 | 1.000 | 0.000 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 5/5 | +0.1165 | [+0.0571, +0.1780] | 0.0604 | 1.000 | 0.000 | - | - |

