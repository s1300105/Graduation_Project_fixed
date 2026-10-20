# Chemistry grid summary

Protocol: docs/CHEM_PROTOCOL_ja.md. Test ROC-AUC, mean +/- sd over seeds; 'best' = epoch with the best validation ROC-AUC, 'final' = last epoch.

## ogb_molbace

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 3 | 0.855 +/- 0.002 | 0.855 +/- 0.002 | 0.866 +/- 0.001 | 0.719 +/- 0.004 |
| A2 graph only | 3 | 0.684 +/- 0.017 | 0.684 +/- 0.017 | 0.663 +/- 0.017 | 0.568 +/- 0.021 |
| A2e graph, no edges | 3 | 0.623 +/- 0.022 | 0.694 +/- 0.012 | 0.625 +/- 0.018 | 0.587 +/- 0.013 |
| A1 text only | 3 | 0.750 +/- 0.010 | 0.792 +/- 0.002 | 0.755 +/- 0.011 | 0.749 +/- 0.004 |
| A3 hybrid | 3 | 0.776 +/- 0.013 | 0.792 +/- 0.014 | 0.796 +/- 0.008 | 0.771 +/- 0.010 |
| A3e hybrid, no edges | 3 | 0.771 +/- 0.006 | 0.789 +/- 0.016 | 0.792 +/- 0.005 | 0.767 +/- 0.011 |
| A3s hybrid, shuffled edges | 2 | 0.776 +/- 0.009 | 0.790 +/- 0.023 | 0.796 +/- 0.005 | 0.768 +/- 0.011 |

## ogb_molbbbp

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 2 | 0.679 +/- 0.002 | 0.679 +/- 0.002 | 0.716 +/- 0.006 | 0.958 +/- 0.002 |
| A2 graph only | 2 | 0.596 +/- 0.006 | 0.592 +/- 0.013 | 0.592 +/- 0.010 | 0.925 +/- 0.003 |
| A2e graph, no edges | 2 | 0.628 +/- 0.000 | 0.628 +/- 0.000 | 0.669 +/- 0.004 | 0.910 +/- 0.003 |
| A1 text only | 2 | 0.664 +/- 0.002 | 0.700 +/- 0.008 | 0.706 +/- 0.005 | 0.959 +/- 0.000 |
| A3 hybrid | 2 | 0.675 +/- 0.009 | 0.706 +/- 0.013 | 0.720 +/- 0.004 | 0.958 +/- 0.000 |
| A3e hybrid, no edges | 2 | 0.666 +/- 0.036 | 0.717 +/- 0.006 | 0.705 +/- 0.026 | 0.958 +/- 0.002 |
| A3s hybrid, shuffled edges | 2 | 0.676 +/- 0.014 | 0.727 +/- 0.007 | 0.727 +/- 0.001 | 0.960 +/- 0.003 |

### ogb_molbace: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 3/3 | +0.0254 | [-0.0051, +0.0546] | 0.0299 | 0.951 | 0.098 | 0.392 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 3/3 | +0.0254 | [+0.0028, +0.0490] | 0.0231 | 0.986 | 0.028 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 3/3 | +0.0002 | [-0.0247, +0.0234] | 0.0241 | 0.496 | 0.992 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 3/3 | +0.0002 | [-0.0152, +0.0162] | 0.0157 | 0.501 | 0.998 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 3/3 | +0.0042 | [-0.0198, +0.0240] | 0.0219 | 0.695 | 0.610 | 1.000 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 3/3 | +0.0042 | [-0.0046, +0.0130] | 0.0088 | 0.834 | 0.332 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 3/3 | +0.0024 | [-0.0217, +0.0281] | 0.0249 | 0.571 | 0.858 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 3/3 | +0.0024 | [-0.0035, +0.0087] | 0.0061 | 0.780 | 0.440 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 3/2 | +0.0000 | [-0.0257, +0.0204] | 0.0231 | 0.536 | 0.928 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 3/2 | +0.0000 | [-0.0112, +0.0113] | 0.0112 | 0.507 | 0.986 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 3/2 | +0.0021 | [-0.0282, +0.0328] | 0.0305 | 0.553 | 0.894 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 3/2 | +0.0021 | [-0.0050, +0.0105] | 0.0077 | 0.718 | 0.564 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 3/3 | +0.0603 | [-0.0086, +0.1324] | 0.0705 | 0.953 | 0.094 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 3/3 | +0.0603 | [-0.0043, +0.1215] | 0.0629 | 0.964 | 0.072 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 3/3 | -0.0102 | [-0.0943, +0.0661] | 0.0802 | 0.368 | 0.736 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 3/3 | -0.0102 | [-0.0864, +0.0627] | 0.0746 | 0.375 | 0.750 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 3/3 | -0.0796 | [-0.1390, -0.0267] | 0.0562 | 0.002 | 0.004 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 3/3 | -0.0796 | [-0.1407, -0.0264] | 0.0571 | 0.003 | 0.006 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 3/3 | -0.0636 | [-0.1190, -0.0122] | 0.0534 | 0.005 | 0.010 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 3/3 | -0.0636 | [-0.1174, -0.0135] | 0.0520 | 0.005 | 0.010 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 3/3 | -0.1050 | [-0.1728, -0.0443] | 0.0643 | 0.001 | 0.002 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 3/3 | -0.1050 | [-0.1708, -0.0460] | 0.0624 | 0.001 | 0.002 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 3/3 | -0.0637 | [-0.1151, -0.0143] | 0.0504 | 0.004 | 0.008 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 3/3 | -0.0637 | [-0.1163, -0.0151] | 0.0506 | 0.005 | 0.010 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 3/3 | +0.0920 | [+0.0122, +0.1836] | 0.0857 | 0.988 | 0.024 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 3/3 | +0.0920 | [+0.0165, +0.1797] | 0.0816 | 0.991 | 0.018 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 3/3 | +0.1081 | [+0.0312, +0.1977] | 0.0832 | 0.996 | 0.008 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 3/3 | +0.1081 | [+0.0334, +0.1940] | 0.0803 | 0.997 | 0.006 | - | - |

### ogb_molbbbp: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 2/2 | +0.0109 | [-0.0279, +0.0480] | 0.0379 | 0.726 | 0.548 | 1.000 | no difference shown |
| primary | A3 hybrid - A1 text only | best | paired_items | 2/2 | +0.0109 | [-0.0206, +0.0433] | 0.0319 | 0.747 | 0.506 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 2/2 | +0.0058 | [-0.0448, +0.0531] | 0.0490 | 0.576 | 0.848 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 2/2 | +0.0058 | [-0.0388, +0.0501] | 0.0445 | 0.579 | 0.842 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | best | two_level | 2/2 | +0.0081 | [-0.0349, +0.0551] | 0.0450 | 0.633 | 0.734 | 1.000 | no difference shown |
| primary | A3 hybrid - A3e hybrid, no edges | best | paired_items | 2/2 | +0.0081 | [-0.0111, +0.0278] | 0.0194 | 0.788 | 0.424 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | two_level | 2/2 | -0.0115 | [-0.0433, +0.0119] | 0.0276 | 0.168 | 0.336 | - | - |
| primary | A3 hybrid - A3e hybrid, no edges | final | paired_items | 2/2 | -0.0115 | [-0.0273, +0.0033] | 0.0153 | 0.074 | 0.148 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | two_level | 2/2 | -0.0015 | [-0.0289, +0.0224] | 0.0256 | 0.468 | 0.936 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | best | paired_items | 2/2 | -0.0015 | [-0.0106, +0.0084] | 0.0095 | 0.381 | 0.762 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | two_level | 2/2 | -0.0217 | [-0.0502, +0.0045] | 0.0273 | 0.043 | 0.086 | - | - |
| secondary | A3 hybrid - A3s hybrid, shuffled edges | final | paired_items | 2/2 | -0.0217 | [-0.0378, -0.0076] | 0.0151 | 0.001 | 0.002 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 2/2 | -0.0314 | [-0.0754, +0.0130] | 0.0442 | 0.088 | 0.176 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 2/2 | -0.0314 | [-0.0764, +0.0103] | 0.0433 | 0.069 | 0.138 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 2/2 | -0.0356 | [-0.0820, +0.0099] | 0.0460 | 0.066 | 0.132 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 2/2 | -0.0356 | [-0.0823, +0.0071] | 0.0447 | 0.051 | 0.102 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 2/2 | -0.0046 | [-0.0665, +0.0534] | 0.0600 | 0.433 | 0.866 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 2/2 | -0.0046 | [-0.0634, +0.0533] | 0.0583 | 0.438 | 0.876 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 2/2 | +0.0267 | [-0.0330, +0.0831] | 0.0580 | 0.798 | 0.404 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 2/2 | +0.0267 | [-0.0324, +0.0838] | 0.0581 | 0.816 | 0.368 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 2/2 | -0.0155 | [-0.0689, +0.0389] | 0.0539 | 0.276 | 0.552 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 2/2 | -0.0155 | [-0.0679, +0.0362] | 0.0520 | 0.277 | 0.554 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 2/2 | +0.0209 | [-0.0292, +0.0699] | 0.0496 | 0.800 | 0.400 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 2/2 | +0.0209 | [-0.0262, +0.0673] | 0.0467 | 0.804 | 0.392 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 2/2 | +0.0782 | [+0.0247, +0.1337] | 0.0545 | 0.999 | 0.002 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 2/2 | +0.0782 | [+0.0274, +0.1306] | 0.0516 | 1.000 | 0.000 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 2/2 | +0.1138 | [+0.0483, +0.1753] | 0.0635 | 1.000 | 0.000 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 2/2 | +0.1138 | [+0.0533, +0.1763] | 0.0615 | 1.000 | 0.000 | - | - |

