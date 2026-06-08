# Annexe — Table des 33 classes

Référence pour les figures qui n'affichent que les numéros de classe à 3 chiffres.
Précision = ensemble uniforme (toutes familles disponibles, single-view) sur `val_dir`.

## Triée par n° de classe

| N° | Nom de la classe | Train | Val | Acc. ensemble | |
|---:|:---|---:|---:|---:|:---:|
| **000** | Closing something | 1068 | 228 | 0.575 | 🟠 |
| **001** | Covering something with something | 2727 | 417 | 0.736 | 🟡 |
| **002** | Dropping something into something | 903 | 178 | 0.534 | 🟠 |
| **003** | Folding something | 972 | 285 | 0.761 | 🟡 |
| **004** | Hitting something with something | 1738 | 235 | 0.536 | 🟠 |
| **005** | Holding something | 1459 | 197 | 0.558 | 🟠 |
| **006** | Moving something away from something | 910 | 183 | 0.716 | 🟡 |
| **007** | Moving something closer to something | 907 | 213 | 0.798 | 🟡 |
| **008** | Moving something down | 2741 | 311 | 0.614 | 🟠 |
| **009** | Moving something up | 3170 | 359 | 0.543 | 🟠 |
| **010** | Opening something | 1253 | 332 | 0.617 | 🟠 |
| **011** | Picking something up | 980 | 199 | 0.116 | 🔴 |
| **012** | Pouring something into something | 873 | 278 | 0.788 | 🟡 |
| **013** | Pouring something out of something | 314 | 79 | 0.696 | 🟡 |
| **014** | Pretending to pick something up | 1547 | 228 | 0.544 | 🟠 |
| **015** | Pretending to pour something out of something but something  | 314 | 56 | 0.554 | 🟠 |
| **016** | Pretending to put something into something | 1044 | 68 | 0.118 | 🔴 |
| **017** | Pretending to throw something | 915 | 47 | 0.383 | 🔴 |
| **018** | Pulling something from left to right | 1555 | 169 | 0.704 | 🟡 |
| **019** | Pulling something from right to left | 1587 | 125 | 0.688 | 🟡 |
| **020** | Putting something behind something | 1204 | 127 | 0.559 | 🟠 |
| **021** | Putting something in front of something | 837 | 135 | 0.622 | 🟠 |
| **022** | Putting something into something | 2188 | 292 | 0.592 | 🟠 |
| **023** | Putting something next to something | 2031 | 203 | 0.685 | 🟡 |
| **024** | Putting something onto something | 1608 | 139 | 0.453 | 🟠 |
| **025** | Showing something to the camera | 709 | 261 | 0.410 | 🟠 |
| **026** | Spilling something next to something | 162 | 60 | 0.400 | 🟠 |
| **027** | (absent du train) | 0 | 0 | — | — |
| **028** | Taking something out of something | 1699 | 239 | 0.753 | 🟡 |
| **029** | Throwing something | 2254 | 184 | 0.663 | 🟡 |
| **030** | Turning something upside down | 2058 | 391 | 0.790 | 🟡 |
| **031** | Uncovering something | 2426 | 312 | 0.705 | 🟡 |
| **032** | Unfolding something | 840 | 215 | 0.670 | 🟡 |
| | **Total** | **44993** | **6745** | **moy. 0.590** | |

## Triée par précision (plus facile → plus dur)

| N° | Nom | Acc. | | Train |
|---:|:---|---:|:---:|---:|
| 027 | (absent du train) | — | — | 0 |
| 007 | Moving something closer to something | 0.798 | 🟡 | 907 |
| 030 | Turning something upside down | 0.790 | 🟡 | 2058 |
| 012 | Pouring something into something | 0.788 | 🟡 | 873 |
| 003 | Folding something | 0.761 | 🟡 | 972 |
| 028 | Taking something out of something | 0.753 | 🟡 | 1699 |
| 001 | Covering something with something | 0.736 | 🟡 | 2727 |
| 006 | Moving something away from something | 0.716 | 🟡 | 910 |
| 031 | Uncovering something | 0.705 | 🟡 | 2426 |
| 018 | Pulling something from left to right | 0.704 | 🟡 | 1555 |
| 013 | Pouring something out of something | 0.696 | 🟡 | 314 |
| 019 | Pulling something from right to left | 0.688 | 🟡 | 1587 |
| 023 | Putting something next to something | 0.685 | 🟡 | 2031 |
| 032 | Unfolding something | 0.670 | 🟡 | 840 |
| 029 | Throwing something | 0.663 | 🟡 | 2254 |
| 021 | Putting something in front of something | 0.622 | 🟠 | 837 |
| 010 | Opening something | 0.617 | 🟠 | 1253 |
| 008 | Moving something down | 0.614 | 🟠 | 2741 |
| 022 | Putting something into something | 0.592 | 🟠 | 2188 |
| 000 | Closing something | 0.575 | 🟠 | 1068 |
| 020 | Putting something behind something | 0.559 | 🟠 | 1204 |
| 005 | Holding something | 0.558 | 🟠 | 1459 |
| 015 | Pretending to pour something out of something but something  | 0.554 | 🟠 | 314 |
| 014 | Pretending to pick something up | 0.544 | 🟠 | 1547 |
| 009 | Moving something up | 0.543 | 🟠 | 3170 |
| 004 | Hitting something with something | 0.536 | 🟠 | 1738 |
| 002 | Dropping something into something | 0.534 | 🟠 | 903 |
| 024 | Putting something onto something | 0.453 | 🟠 | 1608 |
| 025 | Showing something to the camera | 0.410 | 🟠 | 709 |
| 026 | Spilling something next to something | 0.400 | 🟠 | 162 |
| 017 | Pretending to throw something | 0.383 | 🔴 | 915 |
| 016 | Pretending to put something into something | 0.118 | 🔴 | 1044 |
| 011 | Picking something up | 0.116 | 🔴 | 980 |

## Légende

- 🟢 ≥ 0.85 — classe "facile" pour l'ensemble
- 🟡 0.65–0.85 — classe "moyenne"
- 🟠 0.40–0.65 — classe "difficile"
- 🔴 < 0.40 — classe "très difficile" (souvent confondue avec sa voisine sémantique)
- — — pas évaluée (0 exemple dans `val_dir`)
