# Controlled weighting experiment (Section 6.4) — results

Budget: 3 s per epsilon-point, 2 epsilon-points, earliest-start value selector, single Choco harness; only the increment rule varies.

## JSSP (n=23 instances, paired)
| Selector | Closure % | Nodes | Fails | CPU (s) | b̄ | ε=1-b̄ |
|---|---|---|---|---|---|---|
| dom/wdeg2004 (scope-sharing) | 5.4 | 70011 | 69493 | 11.12 | 0.171 | 0.829 |
| Per-variable wdeg (AbsCon) | 5.4 | 79799 | 79263 | 11.21 | 0.203 | 0.797 |
| Scope-wdeg2004 + HD | 4.3 | 59107 | 58603 | 11.00 | 0.188 | 0.812 |

## RCPSP (n=70 instances, paired)
| Selector | Closure % | Nodes | Fails | CPU (s) | b̄ | ε=1-b̄ |
|---|---|---|---|---|---|---|
| dom/wdeg2004 (scope-sharing) | 62.9 | 30463 | 30413 | 4.83 | 0.300 | 0.700 |
| Per-variable wdeg (AbsCon) | 64.6 | 33918 | 33873 | 4.83 | 0.334 | 0.666 |
| Scope-wdeg2004 + HD | 65.7 | 26975 | 26927 | 4.62 | 0.312 | 0.688 |
