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
| A1 text only | 4 | 0.667 +/- 0.011 | 0.700 +/- 0.004 | 0.707 +/- 0.004 | 0.958 +/- 0.002 |
| A3 hybrid | 4 | 0.672 +/- 0.006 | 0.708 +/- 0.009 | 0.723 +/- 0.007 | 0.960 +/- 0.002 |
| A3e hybrid, no edges | 4 | 0.666 +/- 0.021 | 0.717 +/- 0.004 | 0.712 +/- 0.017 | 0.958 +/- 0.001 |
| A3s hybrid, shuffled edges | 4 | 0.673 +/- 0.009 | 0.713 +/- 0.017 | 0.725 +/- 0.003 | 0.960 +/- 0.002 |

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
| primary | A3 hybrid - A1 text only | best | two_level | 4/4 | +0.0056 | [-0.0293, +0.0384] | 0.0338 | 0.621 | 0.758 | 1.000 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 4/4 | +0.0056 | [-0.0227, +0.0347] | 0.0287 | 0.639 | 0.722 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 4/4 | +0.0084 | [-0.0370, +0.0538] | 0.0454 | 0.621 | 0.758 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 4/4 | +0.0084 | [-0.0342, +0.0526] | 0.0434 | 0.619 | 0.762 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 4/4 | +0.0065 | [-0.0185, +0.0365] | 0.0275 | 0.698 | 0.604 | 1.000 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 4/4 | +0.0065 | [-0.0049, +0.0176] | 0.0112 | 0.864 | 0.272 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 4/4 | -0.0088 | [-0.0274, +0.0071] | 0.0173 | 0.137 | 0.274 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 4/4 | -0.0088 | [-0.0185, -0.0003] | 0.0091 | 0.021 | 0.042 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 4/4 | -0.0002 | [-0.0178, +0.0189] | 0.0183 | 0.506 | 0.988 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 4/4 | -0.0002 | [-0.0061, +0.0058] | 0.0060 | 0.465 | 0.930 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 4/4 | -0.0051 | [-0.0287, +0.0179] | 0.0233 | 0.319 | 0.638 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 4/4 | -0.0051 | [-0.0137, +0.0032] | 0.0085 | 0.114 | 0.228 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 5/5 | -0.0335 | [-0.0805, +0.0124] | 0.0464 | 0.076 | 0.152 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 5/5 | -0.0335 | [-0.0823, +0.0110] | 0.0466 | 0.078 | 0.156 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 5/5 | -0.0361 | [-0.0837, +0.0107] | 0.0472 | 0.065 | 0.130 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 5/5 | -0.0361 | [-0.0850, +0.0095] | 0.0473 | 0.061 | 0.122 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 4/5 | -0.0043 | [-0.0638, +0.0529] | 0.0583 | 0.438 | 0.876 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 4/5 | -0.0043 | [-0.0619, +0.0527] | 0.0573 | 0.434 | 0.868 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 4/5 | +0.0314 | [-0.0258, +0.0874] | 0.0566 | 0.852 | 0.296 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 4/5 | +0.0314 | [-0.0270, +0.0891] | 0.0580 | 0.859 | 0.282 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 4/5 | -0.0099 | [-0.0641, +0.0434] | 0.0538 | 0.350 | 0.700 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 4/5 | -0.0099 | [-0.0614, +0.0410] | 0.0512 | 0.346 | 0.692 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 4/5 | +0.0229 | [-0.0240, +0.0723] | 0.0481 | 0.833 | 0.334 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 4/5 | +0.0229 | [-0.0238, +0.0702] | 0.0470 | 0.833 | 0.334 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 4/5 | +0.0764 | [+0.0277, +0.1275] | 0.0499 | 0.999 | 0.002 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 4/5 | +0.0764 | [+0.0290, +0.1297] | 0.0504 | 1.000 | 0.000 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 4/5 | +0.1146 | [+0.0524, +0.1744] | 0.0610 | 1.000 | 0.000 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 4/5 | +0.1146 | [+0.0550, +0.1760] | 0.0605 | 1.000 | 0.000 | - | - |

