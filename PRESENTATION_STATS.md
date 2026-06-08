# Presentation — analyses statistiques (extended)

Tous les chiffres viennent des softmax cachés et des labels `val_dir`.
Datasets: train 44 993 / val 6 745 / test 6 913 clips, 33 classes (000–032, 027 absente du train).

## 1. Déséquilibre des classes (train, val)

Les volumes par classe varient drastiquement. La classe 027 est totalement absente du train ⇒ aucun modèle ne peut la prédire correctement sans extrapolation. Cette asymétrie ↔ accuracy par classe est analysée en §10.

**Top 5 + Bottom 5 (train counts):**

| Classe | Nom | Train | Val | % train | % val |
|:---|---:|---:|---:|---:|---:|
| 009 | Moving something up | 3170 | 359 | 7.05% | 5.32% |
| 008 | Moving something down | 2741 | 311 | 6.09% | 4.61% |
| 001 | Covering something with something | 2727 | 417 | 6.06% | 6.18% |
| 031 | Uncovering something | 2426 | 312 | 5.39% | 4.63% |
| 029 | Throwing something | 2254 | 184 | 5.01% | 2.73% |
| 025 | Showing something to the camera | 709 | 261 | 1.58% | 3.87% |
| 013 | Pouring something out of something | 314 | 79 | 0.70% | 1.17% |
| 015 | Pretending to pour something out of s… | 314 | 56 | 0.70% | 0.83% |
| 026 | Spilling something next to something | 162 | 60 | 0.36% | 0.89% |
| 027 | 027 | 0 | 0 | 0.00% | 0.00% |

**Stats train** : min=162, max=3170, médiane=1228, σ=732, ratio max/min=19.6, classes avec 0 exemple=1.

**KL(train || val) = 0.0877**, **KL(val || train) = 0.0895** (faibles ⇒ distributions très proches ; train est un proxy raisonnable de la distribution test).

## 2. Précision par famille sur `val_dir` (single-view)

Confirme l'argument « diversity wins » : aucune famille seule n'atteint l'accuracy de l'ensemble uniforme.

| Famille | Snapshots | Val top-1 |
|:---|---:|---:|
| VideoMAE-Base K400 4f | 3 | 0.5576 |
| VideoMAE-Base SSv2 4f | 3 | 0.6047 |
| TSM ResNet50 4f | 3 | 0.3364 |
| VideoMAE-Large 4f | 2 | 0.5945 |
| **Uniforme (11 modèles)** | 11 | **0.6222** |

## 3. Précision par classe — où l'ensemble aide le plus (et le moins)

Δ = ensemble − meilleure famille seule. Trié par Δ pour visualiser le gain marginal de la diversité par classe.

**Top 8 gains:**

| Classe | Nom | n_val | Best single | Ensemble | Δ |
|:---|---:|---:|:---|:---|---:|
| 006 | Moving something away from something | 183 | 0.672 | 0.716 | **+0.044** |
| 022 | Putting something into something | 292 | 0.555 | 0.592 | **+0.038** |
| 008 | Moving something down | 311 | 0.582 | 0.614 | **+0.032** |
| 028 | Taking something out of something | 239 | 0.728 | 0.753 | **+0.025** |
| 023 | Putting something next to something | 203 | 0.660 | 0.685 | **+0.025** |
| 019 | Pulling something from right to left | 125 | 0.664 | 0.688 | **+0.024** |
| 007 | Moving something closer to something | 213 | 0.775 | 0.798 | **+0.023** |
| 004 | Hitting something with something | 235 | 0.515 | 0.536 | **+0.021** |

**Bottom 5 (l'ensemble ne dépasse pas la meilleure famille seule) :**

| Classe | Nom | n_val | Best single | Ensemble | Δ |
|:---|---:|---:|:---|:---|---:|
| 000 | Closing something | 228 | 0.627 | 0.575 | -0.053 |
| 025 | Showing something to the camera | 261 | 0.467 | 0.410 | -0.057 |
| 016 | Pretending to put something into some… | 68 | 0.191 | 0.118 | -0.074 |
| 011 | Picking something up | 199 | 0.196 | 0.116 | -0.080 |
| 026 | Spilling something next to something | 60 | 0.500 | 0.400 | -0.100 |

## 4. Confusions les plus fréquentes (ensemble sur val_dir)

| Vraie | → Prédit | Erreurs | Vrai vs prédit (verbe) |
|:---|:---|---:|:---|
| 011 Picking something up | 014 Pretending to pick someth… | 56 | Picking → Pretending |
| 009 Moving something up | 014 Pretending to pick someth… | 44 | Moving → Pretending |
| 016 Pretending to put somethi… | 022 Putting something into so… | 44 | Pretending → Putting |
| 002 Dropping something into s… | 022 Putting something into so… | 42 | Dropping → Putting |
| 025 Showing something to the … | 029 Throwing something | 37 | Showing → Throwing |
| 008 Moving something down | 005 Holding something | 30 | Moving → Holding |
| 000 Closing something | 010 Opening something | 29 | Closing → Opening |
| 009 Moving something up | 030 Turning something upside … | 26 | Moving → Turning |
| 029 Throwing something | 017 Pretending to throw somet… | 26 | Throwing → Pretending |
| 032 Unfolding something | 003 Folding something | 26 | Unfolding → Folding |
| 003 Folding something | 032 Unfolding something | 24 | Folding → Unfolding |
| 011 Picking something up | 009 Moving something up | 24 | Picking → Moving |

## 5. Matrice de confusion 33×33 (CSV — copier pour heatmap)

Lignes = vraie classe, colonnes = prédiction de l'ensemble. Bloc CSV directement plottable.

```csv
true\pred,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32
0,131,2,2,6,6,5,0,1,2,2,29,0,0,0,2,0,0,1,0,2,2,0,7,1,4,1,0,0,12,2,6,2,0
1,8,307,2,4,9,2,2,5,0,1,1,4,0,0,7,0,0,0,0,1,4,9,10,3,13,2,0,0,4,3,3,10,3
2,1,0,95,0,2,2,0,0,0,1,2,0,5,0,0,0,1,1,0,0,10,2,42,2,1,0,0,0,7,3,1,0,0
3,3,2,0,217,0,3,0,0,1,0,2,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,4,3,23,24
4,1,15,4,0,126,5,1,4,1,5,1,0,2,1,3,0,0,2,1,0,14,1,6,11,14,2,0,0,1,9,1,3,1
5,0,5,1,1,6,110,1,0,4,2,4,4,1,0,13,0,0,4,0,0,2,1,3,2,2,2,0,0,2,6,14,5,2
6,0,0,1,0,3,1,131,6,1,2,1,4,0,0,3,0,0,0,6,7,0,0,0,2,0,1,1,0,7,0,5,1,0
7,1,2,0,0,2,0,7,170,0,2,1,1,0,0,0,0,0,0,2,4,0,0,1,12,0,1,0,0,0,1,3,1,2
8,2,7,0,1,4,30,0,0,191,3,2,5,0,0,8,0,0,8,1,5,2,7,0,4,16,0,0,0,1,6,5,3,0
9,4,5,0,1,3,13,0,2,3,195,1,21,0,0,44,0,0,4,1,12,0,0,0,0,1,0,0,0,9,9,26,3,2
10,18,0,1,2,2,9,3,2,1,9,205,4,1,1,6,2,0,2,0,1,0,1,4,0,1,3,0,0,16,3,18,12,5
11,0,14,0,0,6,2,3,1,5,24,4,23,0,0,56,0,0,0,1,5,1,4,3,4,3,1,0,0,6,1,22,10,0
12,2,1,6,0,0,1,0,0,0,1,2,0,219,13,0,11,0,0,0,0,6,1,7,2,2,1,2,0,0,0,1,0,0
13,0,0,0,0,0,1,0,0,0,0,0,0,12,55,0,9,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0
14,0,20,0,0,9,2,4,1,6,5,4,12,0,0,124,0,0,0,2,7,2,1,1,4,1,1,0,0,5,0,9,8,0
15,0,0,2,0,1,1,0,0,1,0,0,0,7,3,0,31,0,0,0,0,2,0,2,0,0,0,0,0,1,0,5,0,0
16,0,0,4,0,3,0,0,0,2,0,0,0,1,0,0,0,8,0,0,0,2,0,44,0,3,0,0,0,1,0,0,0,0
17,0,0,0,1,1,2,0,0,0,2,0,1,0,0,0,1,0,18,1,0,1,0,0,0,0,1,0,0,0,16,1,1,0
18,2,0,1,1,1,1,1,6,0,5,1,1,0,0,7,0,0,1,119,0,0,0,0,10,2,0,0,0,1,1,5,2,1
19,1,3,0,0,0,0,4,1,0,3,0,0,0,0,3,0,0,0,1,86,0,0,0,13,0,0,0,0,2,1,1,5,1
20,1,7,4,0,7,0,0,4,0,1,0,1,0,0,2,0,0,0,0,0,71,1,9,10,3,0,0,0,2,1,1,2,0
21,1,23,0,0,1,0,1,5,4,0,0,1,1,0,1,0,0,1,0,0,1,84,3,5,1,1,0,0,0,0,1,0,0
22,4,5,12,1,7,1,6,4,2,4,1,1,6,0,1,1,12,0,0,0,9,0,173,7,11,0,0,0,14,3,5,2,0
23,0,18,0,0,4,1,2,4,1,0,0,1,0,0,1,0,1,2,1,0,7,5,5,139,3,4,0,0,1,1,1,1,0
24,1,14,1,0,3,1,2,0,2,1,0,3,0,0,1,1,0,0,0,0,6,0,18,9,63,3,0,0,1,2,0,7,0
25,3,9,0,1,10,20,1,1,1,17,4,1,1,0,4,0,0,3,1,1,5,1,9,7,8,107,0,0,2,37,6,1,0
26,0,2,1,0,0,1,0,5,0,0,0,0,7,1,0,0,0,0,0,0,1,0,2,9,0,2,24,0,0,4,1,0,0
27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
28,4,1,2,1,2,2,1,1,2,0,9,1,1,0,6,0,0,0,0,1,0,1,10,1,1,2,0,0,180,0,4,5,1
29,1,2,1,2,6,4,0,1,3,4,0,0,0,0,0,0,0,26,0,0,0,1,2,1,1,0,0,0,0,122,5,2,0
30,0,1,0,1,0,4,0,1,7,8,4,4,1,0,14,4,0,1,1,5,1,0,3,1,0,2,0,0,1,2,309,14,2
31,2,8,1,13,5,1,0,0,3,0,13,6,1,0,10,0,0,0,2,2,1,0,2,0,1,2,0,0,3,2,5,220,9
32,1,2,0,26,0,4,0,1,1,2,4,0,0,0,3,0,0,2,0,0,0,0,0,1,0,0,0,0,2,1,6,15,144
```

## 6. Analyse par groupe sémantique (verbe d'action)

Regroupement par verbe d'action (premier mot du nom de classe). Mesure l'accuracy moyenne par type d'action.

| Verbe | # classes | n_train | n_val | Ensemble acc |
|:---|:---|---:|---:|:---|
| Moving | 4 | 7728 | 1066 | 0.644 |
| Putting | 5 | 7868 | 896 | 0.592 |
| Covering | 1 | 2727 | 417 | 0.736 |
| Pretending | 4 | 3820 | 399 | 0.454 |
| Turning | 1 | 2058 | 391 | 0.790 |
| Pouring | 2 | 1187 | 357 | 0.768 |
| Opening | 1 | 1253 | 332 | 0.617 |
| Uncovering | 1 | 2426 | 312 | 0.705 |
| Pulling | 2 | 3142 | 294 | 0.697 |
| Folding | 1 | 972 | 285 | 0.761 |
| Showing | 1 | 709 | 261 | 0.410 |
| Taking | 1 | 1699 | 239 | 0.753 |
| Hitting | 1 | 1738 | 235 | 0.536 |
| Closing | 1 | 1068 | 228 | 0.575 |
| Unfolding | 1 | 840 | 215 | 0.670 |
| Picking | 1 | 980 | 199 | 0.116 |
| Holding | 1 | 1459 | 197 | 0.558 |
| Throwing | 1 | 2254 | 184 | 0.663 |
| Dropping | 1 | 903 | 178 | 0.534 |
| Spilling | 1 | 162 | 60 | 0.400 |

## 7. Analyse des paires miroir (label-aware hflip)

Le pipeline détecte 1 paire miroir : **018 Pulling left→right** ↔ **019 Pulling right→left**. 
Le flip horizontal échange visuellement les deux ⇒ le label est aussi échangé pour rester correct. 
On vérifie ici que la fix supprime bien la confusion entre ces deux classes.

| Vraie classe | → prédit dans mirroir | → prédit ailleurs | Bien classé |
|:---|:---|:---|:---|
| 018 | 0 | 50 | 119 |
| 019 | 1 | 38 | 86 |

Dans val_dir, classe 18 a 169 clips, classe 19 a 125.

## 8. Top-1 vs Top-5 par classe — "presque correct" ?

Quand le top-1 se trompe, est-ce que la vraie classe est souvent dans le top-5 (modèle « presque » bon) ?

**Global** : top-1 = 0.622, top-5 = 0.882, gap = 0.260.

**Top 8 classes où le top-5 récupère le plus** (modèle se trompe top-1 mais sait que c'est proche) :

| Classe | Nom | n_val | Top-1 | Top-5 | Δ (récup.) |
|:---|---:|---:|---:|---:|---:|
| 016 | Pretending to put something into some… | 68 | 0.118 | 0.794 | **+0.676** |
| 011 | Picking something up | 199 | 0.116 | 0.729 | **+0.613** |
| 017 | Pretending to throw something | 47 | 0.383 | 0.851 | **+0.468** |
| 014 | Pretending to pick something up | 228 | 0.544 | 0.899 | **+0.355** |
| 024 | Putting something onto something | 139 | 0.453 | 0.799 | **+0.345** |
| 002 | Dropping something into something | 178 | 0.534 | 0.860 | **+0.326** |
| 004 | Hitting something with something | 235 | 0.536 | 0.860 | **+0.323** |
| 015 | Pretending to pour something out of s… | 56 | 0.554 | 0.875 | **+0.321** |

## 9. Spécialisation par famille de modèles

Pour chaque classe, quelle famille a la meilleure précision ? Donne le nombre de classes "remportées" par famille — illustre la complémentarité.

| Famille | Classes remportées (sur 32 actives) |
|:---|---:|
| VideoMAE-Base SSv2 4f | 17 |
| VideoMAE-Large 4f | 13 |
| TSM ResNet50 4f | 1 |
| VideoMAE-Base K400 4f | 1 |

## 10. Corrélation taille de la classe ↔ précision

Est-ce que les classes avec plus d'exemples d'entraînement sont mieux apprises ?

**Corrélation Pearson** : r = 0.239 (linéaire), r(log) = 0.234.

Données brutes pour scatter plot (train_count, val_acc) :
```csv
class_idx,train_count,val_acc
26,162,0.4000
13,314,0.6962
15,314,0.5536
25,709,0.4100
21,837,0.6222
32,840,0.6698
12,873,0.7878
2,903,0.5337
7,907,0.7981
6,910,0.7158
17,915,0.3830
3,972,0.7614
11,980,0.1156
16,1044,0.1176
0,1068,0.5746
20,1204,0.5591
10,1253,0.6175
5,1459,0.5584
14,1547,0.5439
18,1555,0.7041
19,1587,0.6880
24,1608,0.4532
28,1699,0.7531
4,1738,0.5362
23,2031,0.6847
30,2058,0.7903
22,2188,0.5925
29,2254,0.6630
31,2426,0.7051
1,2727,0.7362
8,2741,0.6141
9,3170,0.5432
```

## 11. Accord pair-à-pair sur le test (diversité)

**Accord moyen intra-famille** (= cohérence interne des snapshots) :

| Famille | Snapshots | Accord intra |
|:---|---:|---:|
| V-JEPA-ft | 3 | 0.897 |
| V-JEPA-pseudo | 3 | 0.893 |
| K400-Base | 3 | 0.883 |
| TSM | 3 | 0.879 |
| Large-4f | 2 | 0.786 |

**Accord moyen inter-familles** (= diversité — plus bas = plus de complémentarité) :

| Famille A | Famille B | Accord moyen |
|:---|:---|---:|
| V-JEPA-pseudo | TSM | 0.370 |
| V-JEPA-ft | TSM | 0.372 |
| K400-Base | TSM | 0.375 |
| TSM | VMAE-L-win | 0.382 |
| TSM | Large-4f | 0.398 |
| K400-Base | VMAE-L-win | 0.577 |
| V-JEPA-ft | K400-Base | 0.582 |
| V-JEPA-pseudo | K400-Base | 0.587 |
| V-JEPA-ft | Large-4f | 0.640 |
| K400-Base | Large-4f | 0.640 |
| V-JEPA-pseudo | Large-4f | 0.644 |
| Large-4f | VMAE-L-win | 0.664 |
| V-JEPA-pseudo | VMAE-L-win | 0.688 |
| V-JEPA-ft | VMAE-L-win | 0.688 |
| V-JEPA-ft | V-JEPA-pseudo | 0.854 |

## 12. Couverture des pseudo-labels (seuil 0.85)

**Test clips pseudo-étiquetés** : 6913 / 6913 (100.0%) au seuil de confiance ≥ 0.85.

**Couverture par classe** (top 10) :

| Classe | Nom | n_pseudo | % du test pseudo-étiqueté |
|:---|---:|---:|---:|
| 029 | Throwing something | 5216 | 75.5% |
| 001 | Covering something with something | 357 | 5.2% |
| 004 | Hitting something with something | 273 | 3.9% |
| 019 | Pulling something from right to left | 231 | 3.3% |
| 003 | Folding something | 185 | 2.7% |
| 018 | Pulling something from left to right | 162 | 2.3% |
| 008 | Moving something down | 135 | 2.0% |
| 022 | Putting something into something | 133 | 1.9% |
| 009 | Moving something up | 70 | 1.0% |
| 031 | Uncovering something | 42 | 0.6% |

**Classes jamais pseudo-étiquetées** : 10 / 33 — ces classes ne reçoivent **aucun** renforcement par pseudo-label.

## 13. Distribution de la confiance (ensemble final, test)

**Stats** : moyenne=0.478, médiane=0.451, min=0.071, max=0.944.

**Histogramme** (probabilité max de l'ensemble par clip) :

| Plage | # clips | % |
|:---|:---|---:|
| [0.00, 0.20) | 655 | 9.5% |
| [0.20, 0.30) | 1085 | 15.7% |
| [0.30, 0.40) | 1135 | 16.4% |
| [0.40, 0.50) | 1067 | 15.4% |
| [0.50, 0.60) | 820 | 11.9% |
| [0.60, 0.70) | 792 | 11.5% |
| [0.70, 0.85) | 983 | 14.2% |
| [0.85, 1.00) | 376 | 5.4% |

**Entropie moyenne des prédictions sur val par classe** (haute = modèle indécis sur cette classe) :

Top 5 plus indécises:

| Classe | Nom | n_val | Entropie moy. | Acc |
|:---|---:|---:|:---|---:|
| 026 | Spilling something next to something | 60 | 2.650 | 0.400 |
| 025 | Showing something to the camera | 261 | 2.613 | 0.410 |
| 011 | Picking something up | 199 | 2.524 | 0.116 |
| 017 | Pretending to throw something | 47 | 2.519 | 0.383 |
| 010 | Opening something | 332 | 2.474 | 0.617 |

## 14. Valeur marginale par snapshot (leave-one-out sur val)

Δ = acc(ensemble complet) − acc(ensemble sans ce snapshot). Δ > 0 ⇒ retirer le snapshot **fait baisser** l'ensemble = il apporte vraiment.

| Snapshot | Acc sans lui | Δ |
|:---|---:|---:|
| videomae_ovn1_ssv2_top1 | 0.6147 | +0.0076 |
| videomae_ovn2_large_attempt1_top2 | 0.6160 | +0.0062 |
| videomae_ovn1_ssv2_top2 | 0.6163 | +0.0059 |
| videomae_ovn1_ssv2_top3 | 0.6163 | +0.0059 |
| videomae_ovn2_large_attempt1_top3 | 0.6181 | +0.0042 |
| tsm_r50_ovn1_top2 | 0.6227 | -0.0004 |
| tsm_r50_ovn1_top1 | 0.6228 | -0.0006 |
| tsm_r50_ovn1_top3 | 0.6233 | -0.0010 |
| videomae_ovn1_k400_top2 | 0.6258 | -0.0036 |
| videomae_ovn1_k400_top3 | 0.6276 | -0.0053 |
| videomae_ovn1_k400_top1 | 0.6277 | -0.0055 |

## 15. Distribution prédite sur le test vs distribution réelle val

| Classe | Nom | Val (réel) | Test (prédit) | Δ |
|:---|---:|---:|:---|---:|
| 009 | Moving something up | 5.3% | 4.6% | -0.8 pp |
| 008 | Moving something down | 4.6% | 2.9% | -1.7 pp |
| 001 | Covering something with something | 6.2% | 6.4% | +0.2 pp |
| 031 | Uncovering something | 4.6% | 4.6% | -0.0 pp |
| 029 | Throwing something | 2.7% | 3.5% | +0.7 pp |
| 022 | Putting something into something | 4.3% | 5.1% | +0.7 pp |
| 030 | Turning something upside down | 5.8% | 7.7% | +1.9 pp |
| 023 | Putting something next to something | 3.0% | 3.3% | +0.3 pp |
| 004 | Hitting something with something | 3.5% | 4.2% | +0.7 pp |
| 028 | Taking something out of something | 3.5% | 4.2% | +0.7 pp |

## 16. Récapitulatif des chiffres-clé

- **Accuracy ensemble val_dir** : 0.6222 (11 modèles, uniforme)
- **Meilleure famille seule** : 0.6047
- **Gap top-1 vs top-5** : 0.260 (top-5 = 0.882)
- **Corrélation taille classe ↔ acc** : r = 0.239
- **KL(train‖val)** : 0.0877 — distributions très proches
- **Confiance moy. ensemble test** : 0.478 (médiane 0.451)
- **Pseudo-labels au seuil 0.85** : 6913/6913 (100.0%) clips, 10/33 classes non couvertes
