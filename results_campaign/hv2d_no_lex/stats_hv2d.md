# Analyse statistique — HV 2D normalisé (makespan × flowtime)

> HV recalculé sur les 2 objectifs comparables, normalisés dans [0,1]² par (instance × snapshot) avec idéal/nadir partagés entre heuristiques. L'axe robustesse est exclu (non instrumenté côté solveurs CP — `choco_runner.py` L137).

Blocs (instance × snapshot × seed) : **4075** | heuristiques : **6**

## Rangs moyens (1 = meilleur)

| Heuristique | Rang moyen |
|-------------|-----------:|
| nsga2 | 2.857 |
| nsga3 | 2.870 |
| wdeg | 3.307 |
| mo_dyn_hd_cacd ⭐ | 3.339 |
| dom | 3.607 |
| activity | 5.020 |

## Test de Friedman

- χ² = **4107.476**, p = **0.000e+00**
- Différence globale significative (p < 0.05).

## Post-hoc de Nemenyi (p-values par paires)

```
heuristic       activity  dom  mo_dyn_hd_cacd   nsga2   nsga3    wdeg
heuristic                                                            
activity             1.0  0.0          0.0000  0.0000  0.0000  0.0000
dom                  0.0  1.0          0.0000  0.0000  0.0000  0.0000
mo_dyn_hd_cacd       0.0  0.0          1.0000  0.0000  0.0000  0.9726
nsga2                0.0  0.0          0.0000  1.0000  0.9996  0.0000
nsga3                0.0  0.0          0.0000  0.9996  1.0000  0.0000
wdeg                 0.0  0.0          0.9726  0.0000  0.0000  1.0000
```

## mo_dyn_hd_cacd vs autres — Wilcoxon apparié + Â₁₂

| vs | médiane(ref) | médiane(autre) | Wilcoxon p | Â₁₂ (ref meilleur) | effet |
|----|----:|----:|----:|----:|----|
| activity | 1.0456 | 0.1100 | **0.000e+00** | 0.778 | grand |
| dom | 1.0456 | 1.0337 | **7.602e-46** | 0.489 | négligeable |
| nsga2 | 1.0456 | 1.1386 | **2.039e-07** | 0.467 | négligeable |
| nsga3 | 1.0456 | 1.1326 | **5.060e-07** | 0.468 | négligeable |
| wdeg | 1.0456 | 1.0534 | **8.902e-21** | 0.490 | négligeable |

_p < 0.05 en gras. Â₁₂ > 0.5 ⇒ la référence est meilleure._
