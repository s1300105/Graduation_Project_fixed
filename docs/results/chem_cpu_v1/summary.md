# Chemistry grid summary

Protocol: docs/CHEM_PROTOCOL_ja.md. Test ROC-AUC, mean +/- sd over seeds; 'best' = epoch with the best validation ROC-AUC, 'final' = last epoch.

## ogb_molbbbp

| arm | n | test ROC-AUC (best) | test ROC-AUC (final) | test PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|
| A0 ECFP4+RF | 1 | 0.857 +/- 0.000 | 0.857 +/- 0.000 | 0.866 +/- 0.000 | 0.719 +/- 0.000 |
| A2 graph only | 1 | 0.664 +/- 0.000 | 0.664 +/- 0.000 | 0.644 +/- 0.000 | 0.544 +/- 0.000 |
| A2e graph, no edges | 1 | 0.630 +/- 0.000 | 0.704 +/- 0.000 | 0.631 +/- 0.000 | 0.588 +/- 0.000 |
| A1 text only | 1 | 0.740 +/- 0.000 | 0.789 +/- 0.000 | 0.743 +/- 0.000 | 0.753 +/- 0.000 |
| A3 hybrid | 1 | 0.776 +/- 0.000 | 0.790 +/- 0.000 | 0.794 +/- 0.000 | 0.773 +/- 0.000 |

### ogb_molbace: contrasts (ROC-AUC, A - B; 2000 resamples)

| role | A - B | preds | statistic | n runs | diff | 95% CI | half-width | P(A>B) | p two-sided | p Holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | A3 hybrid - A1 text only | best | two_level | 1/1 | +0.0362 | [+0.0090, +0.0665] | 0.0288 | 0.996 | 0.008 | 0.008 | positive |
| primary | A3 hybrid - A1 text only | best | paired_items | 1/1 | +0.0362 | [+0.0090, +0.0665] | 0.0288 | 0.996 | 0.008 | - | - |
| primary | A3 hybrid - A1 text only | final | two_level | 1/1 | +0.0007 | [-0.0234, +0.0251] | 0.0243 | 0.513 | 0.974 | - | - |
| primary | A3 hybrid - A1 text only | final | paired_items | 1/1 | +0.0007 | [-0.0234, +0.0251] | 0.0243 | 0.513 | 0.974 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | two_level | 1/1 | +0.0341 | [-0.0367, +0.1002] | 0.0684 | 0.816 | 0.368 | - | - |
| secondary | A2 graph only - A2e graph, no edges | best | paired_items | 1/1 | +0.0341 | [-0.0367, +0.1002] | 0.0684 | 0.816 | 0.368 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | two_level | 1/1 | -0.0393 | [-0.1245, +0.0424] | 0.0834 | 0.167 | 0.334 | - | - |
| secondary | A2 graph only - A2e graph, no edges | final | paired_items | 1/1 | -0.0393 | [-0.1245, +0.0424] | 0.0834 | 0.167 | 0.334 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | two_level | 1/1 | -0.0805 | [-0.1432, -0.0239] | 0.0596 | 0.004 | 0.008 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | best | paired_items | 1/1 | -0.0805 | [-0.1432, -0.0239] | 0.0596 | 0.004 | 0.008 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | two_level | 1/1 | -0.0668 | [-0.1256, -0.0144] | 0.0556 | 0.005 | 0.010 | - | - |
| secondary | A3 hybrid - A0 ECFP4+RF | final | paired_items | 1/1 | -0.0668 | [-0.1256, -0.0144] | 0.0556 | 0.005 | 0.010 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | two_level | 1/1 | -0.1167 | [-0.1871, -0.0524] | 0.0673 | 0.000 | 0.000 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | best | paired_items | 1/1 | -0.1167 | [-0.1871, -0.0524] | 0.0673 | 0.000 | 0.000 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | two_level | 1/1 | -0.0675 | [-0.1246, -0.0142] | 0.0552 | 0.007 | 0.014 | - | - |
| exploratory | A1 text only - A0 ECFP4+RF | final | paired_items | 1/1 | -0.0675 | [-0.1246, -0.0142] | 0.0552 | 0.007 | 0.014 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | two_level | 1/1 | +0.1118 | [+0.0327, +0.2045] | 0.0859 | 0.998 | 0.004 | - | - |
| exploratory | A3 hybrid - A2 graph only | best | paired_items | 1/1 | +0.1118 | [+0.0327, +0.2045] | 0.0859 | 0.998 | 0.004 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | two_level | 1/1 | +0.1257 | [+0.0451, +0.2195] | 0.0872 | 0.999 | 0.002 | - | - |
| exploratory | A3 hybrid - A2 graph only | final | paired_items | 1/1 | +0.1257 | [+0.0451, +0.2195] | 0.0872 | 0.999 | 0.002 | - | - |

