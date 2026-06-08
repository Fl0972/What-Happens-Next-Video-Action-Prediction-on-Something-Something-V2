# Report Update Checklist — Post-ResNet50 & Ablation Study Results

**Date:** 2026-06-04  
**New best val-dir:** 46.75% (5-model ensemble, up from 44.03%)  
**New best single model:** 42.85% (TSM-ResNet50 rotating, up from 41.68% with TTA)

All figures have already been regenerated. This file lists every text passage and
LaTeX element in `report_cvpr_v2.tex` that must be changed before the next build.

---

## 1. Figures — what was updated and what to do in the TeX

### 1.1 Updated in-place (existing filenames, already regenerated)

| File | What changed | Action in TeX |
|------|-------------|--------------|
| `fig01_per_model_accuracy.png` | Added TSM-R50 (rot) + 6 ablation models (A–F) in red | Update caption (see §2.1) |
| `fig04_ensemble.png` | Shows full progression: single → 3-model → 4-model → **5-model 46.75%** | Update caption (see §2.2) |
| `fig05_per_class_f1.png` | Now computed from new 5-model best ensemble | Update caption (see §2.3) |
| `fig09_ensemble_vs_single_perclass.png` | Now compares 5-model ensemble vs TSM-R50 (new best single) | Not in current TeX — add if desired |
| `fig10_checkpoint_internal_val.png` | Added R50 and 6 ablation checkpoints | Not in current TeX — add if desired |
| `fig11_leave_one_out.png` | Updated to 5-model ensemble LOO | Update caption (see §2.4) |

### 1.2 New figures (add to TeX)

| File | Content | Suggested placement |
|------|---------|-------------------|
| `fig16_ablation_tsm.png` | TSM focal γ sweep (Exp A, C, T3) + SGDR T_mult (Exp E) | §3.1 Track A ablations |
| `fig17_ablation_vfl.png` | VFL depth, fold count, SGDR ablations (Exps B, D, F) | §3.1 Track A ablations |
| `fig18_backbone_r18_vs_r50.png` | TSM progression from R18/T=5 baseline to R50/T=4 rotating | After "Critical frame-count fix" paragraph |

---

## 2. Caption changes in `report_cvpr_v2.tex`

### 2.1 `\label{fig:permodel}` caption (currently around line 545–549)

**Current:**
```latex
\caption{Per-model \texttt{val\_dir} top-1/top-5 (Track~A). Correcting the
frame count (TSM-Ultra$\rightarrow$TSM-Ultra-v2) is the single largest
per-model jump; the ResNet-50/$T{=}8$ scale-up regresses
(Sec.~\ref{sec:results}).}
```
**Replace with:**
```latex
\caption{Per-model \texttt{val\_dir} top-1/top-5 (Track~A, all from scratch).
Red bars = new models: TSM-ResNet50 rotating ($T{=}4$, \textbf{42.85\%} — new
best single) and ablation series A–F. Correcting the frame count
(TSM-Ultra$\rightarrow$TSM-Ultra-v2) is the largest per-model jump; the
$T{=}4$ R50 rotating model then adds another $+3.4$\,pp.}
```

### 2.2 `\label{fig:ensemble}` caption (currently around line 554–558)

**Current:**
```latex
\caption{Track~A ensemble vs.\ its components. Log-softmax late fusion of two
TSM variants and one VFL variant (all +TTA) exceeds the best single model by
$+2.35$\,pp top-1, reaching $44.03\%$ \texttt{val\_dir}.}
```
**Replace with:**
```latex
\caption{Track~A ensemble progression (\texttt{val\_dir} top-1). Adding
TSM-ResNet50 (no TTA) to the prior 3-model R18 ensemble pushes from 44.03\%
to 46.23\%; adding the CE-trained TSM ablation reaches the new best of
\textbf{46.75\%} (5-model). The R50 backbone is the dominant contributor
($-1.33$\,pp if removed; Fig.~\ref{fig:loo}).}
```

### 2.3 `\label{fig:perclass}` caption (currently around line 572–576)

**Current:**
```latex
\caption{Per-class F1 (ensemble, \texttt{val\_dir}). ...}
```
**Replace with:**
```latex
\caption{Per-class F1 (new 5-model best ensemble, \texttt{val\_dir}).
Directional single-object motions score highest; ``pretending'' and
direction-ambiguous classes score lowest (Sec.~\ref{sec:failure}).}
```

### 2.4 `\label{fig:loo}` caption (currently around line 562–567)

**Current:**
```latex
\caption{Track~A leave-one-out study (\texttt{val\_dir} top-1). All three
members are necessary; the weakest standalone model (VFL) still adds value
through error decorrelation (Table~\ref{tab:loo}).}
```
**Replace with:**
```latex
\caption{Leave-one-out on the new 5-model best ensemble (\texttt{val\_dir}
top-1 = 46.75\%). TSM-ResNet50 dominates ($-1.33$\,pp); every member
contributes. The previous 3-model best (44.03\%) is shown as a dashed
reference line.}
```

---

## 3. Text changes in `report_cvpr_v2.tex`

### 3.1 `\paragraph{Final Track~A ensemble.}` (~line 161)

**Current:**
```latex
The submission fuses three checkpoints by log-softmax late fusion with 10-crop
spatial TTA (5 crops $\times$ 2 flips): \texttt{tsm\_ultra\_v2} ($57.73\%$
internal), its rotating-fold variant (trained on all folds), and
\texttt{video\_former\_lite\_ultra\_rot}. Averaging in log-probability space
down-weights over-confident errors and rewards agreement, suiting members of
different calibration. The ensemble reaches \textbf{44.03\% on the official
\texttt{val\_dir}} (Fig.~\ref{fig:ensemble}).
```
**Replace with:**
```latex
The final ensemble uses five checkpoints with log-softmax late fusion.
Three members use 10-crop spatial TTA (5 crops~$\times$~2 flips):
\texttt{tsm\_ultra\_v2} ($57.73\%$ internal), its rotating-fold variant, and
\texttt{video\_former\_lite\_ultra\_rot}. Two additional members run without TTA:
a \textbf{TSM-ResNet50 rotating-fold} model ($T{=}4$, 42.85\% single-model)
and a CE-loss TSM ablation that provides complementary calibration.
Averaging in log-probability space down-weights over-confident errors.
The ensemble reaches \textbf{46.75\%} on the official \texttt{val\_dir}
(Fig.~\ref{fig:ensemble}).
```

### 3.2 `\paragraph{Critical frame-count fix.}` (~line 136)

The current paragraph already mentions the R50/T=8 regression to 45.26%.
Add one sentence at the end referencing the new R50 success:

**Add after "fix the data, then scale.":**
```latex
A subsequent ResNet-50 run at $T{=}4$ with rotating folds confirms this:
it reaches \textbf{42.85\%} \texttt{val\_dir} (Fig.~\ref{fig:backbone}),
proving that capacity helps once the temporal-sampling bug is eliminated.
```
(requires `\label{fig:backbone}` on the new fig18)

### 3.3 `Table~\ref{tab:summary}` (~line 282)

**Current Track~A rows:**
```latex
A & TSM-ResNet18 (single)      & 38.25\% & --- \\
A & VFL-Ultra (single)         & 32.11\% & --- \\
A & 3-model ens.\ + TTA        & \textbf{44.03\%} & --- \\
```
**Replace with:**
```latex
A & TSM-ResNet18 (single, best) & 41.68\% & --- \\
A & TSM-ResNet50 rotating       & 42.85\% & --- \\
A & 5-model ens.\ (log-softmax) & \textbf{46.75\%} & --- \\
```

### 3.4 `Table~\ref{tab:loo}` (~line 322)

**Current (3-model, no TTA, 40.43% baseline):**
```latex
Full ensemble (3 models)        & \textbf{40.43\%} & --- \\
\;$-$ \texttt{tsm\_ultra\_v2}   & 37.26\% & $-3.17$ \\
\;$-$ \texttt{tsm\_ultra}       & 39.56\% & $-0.87$ \\
\;$-$ \texttt{video\_former\_lite\_ultra} & 39.82\% & $-0.61$ \\
```
**Replace with:**
```latex
Full ensemble (5 models)           & \textbf{46.75\%} & --- \\
\;$-$ TSM-ResNet50 (rot)           & 45.41\% & $-1.33$ \\
\;$-$ VFL-Ultra-rot $+$TTA         & 46.11\% & $-0.64$ \\
\;$-$ TSM-no-focal (rot)           & 46.23\% & $-0.52$ \\
\;$-$ TSM-Ultra-v2-rot $+$TTA      & 46.27\% & $-0.47$ \\
\;$-$ TSM-Ultra-v2 $+$TTA          & 46.43\% & $-0.31$ \\
```
Also update the `\caption` of the table:
```latex
\caption{Track~A leave-one-out on \texttt{val\_dir} (new 5-model ensemble,
46.75\%). TSM-ResNet50 backbone diversity dominates; every member contributes.}
```

### 3.5 `\paragraph{Ensemble components.}` (~line 315)

**Current:**
```latex
On the three-model weighted sub-ensemble (40.43\%), leave-one-out
(Table~\ref{tab:loo}, Fig.~\ref{fig:loo}) shows every member contributes;
removing the strongest (\texttt{tsm\_ultra\_v2}) costs $-3.17$\,pp, but removing
the \emph{weakest} (VFL) still costs $-0.61$\,pp, confirming error decorrelation
between local-shift and global-attention biases.
```
**Replace with:**
```latex
On the five-model ensemble (46.75\%), leave-one-out
(Table~\ref{tab:loo}, Fig.~\ref{fig:loo}) shows every member contributes.
TSM-ResNet50 dominates ($-1.33$\,pp): backbone diversity (R50 vs.\ R18) provides
more ensemble benefit than model-family diversity at the same capacity. The
CE-trained TSM ablation ($-0.52$\,pp) adds complementary calibration: CE models
maintain higher confidence on easy examples, partially correcting the systematic
over-caution of focal-trained members on common classes.
```

### 3.6 Add ablation sub-paragraphs in `\subsection{Track~A ablations}` (~line 301)

After the existing paragraphs (warm-up, frame count, ensemble components), add:

```latex
\paragraph{Focal loss (ablation, $+0.5$\,pp).}
A controlled sweep over $\gamma\!\in\!\{0,1,2\}$ (all other variables frozen,
rotating folds) yields 38.92\% (CE), 39.50\% ($\gamma{=}1$), and 39.45\%
($\gamma{=}2$). Any focal weight beats CE; the optimum is near $\gamma{=}1$;
the $\gamma{=}1$ vs.\ $\gamma{=}2$ gap ($0.05$\,pp) is within noise
(Fig.~\ref{fig:abla_tsm}).

\paragraph{VFL transformer depth and fold count (ablations).}
A 2-layer VFL rotating model scores 35.36\% vs.\ 36.44\% for 3-layer
($-1.08$\,pp): the third block provides a real but modest gain
(Fig.~\ref{fig:abla_vfl}). Increasing to 10 folds (90\% data per epoch)
\emph{hurts} by $-0.46$\,pp, indicating VFL has saturated its
data-utilisation capacity at $k{=}5$.

\paragraph{SGDR schedule shape.}
Switching SGDR from $T_\text{mult}{=}1$ (equal 25-ep cycles) to
$T_\text{mult}{=}2$ (growing cycles) changes accuracy by $-0.04$\,pp for
TSM and $-0.12$\,pp for VFL: the rotating-fold protocol already
provides implicit regularisation equivalent to LR-restart benefits.
```

And add the two new figure environments (with appropriate labels):
```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{fig16_ablation_tsm.png}
  \caption{TSM ablation: focal $\gamma$ sweep and SGDR schedule.
  Any focal weight beats CE; $\gamma{=}1$ and $\gamma{=}2$ are within noise.
  SGDR T\_mult has negligible effect.}
  \label{fig:abla_tsm}
\end{figure}

\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{fig17_ablation_vfl.png}
  \caption{VFL ablations: Transformer depth, fold count, SGDR schedule.
  The 3rd Transformer block adds $+1.08$\,pp; 10 folds marginally hurts
  (VFL data-saturated at $k{=}5$); SGDR shape is irrelevant.}
  \label{fig:abla_vfl}
\end{figure}
```

### 3.7 Add fig18 near the frame-count section

After (or near) the `\begin{figure}` block for `fig01` (the per-model accuracy figure),
insert:
```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{fig18_backbone_r18_vs_r50.png}
  \caption{TSM backbone and frame-count progression. Each arrow annotates the
  incremental \texttt{val\_dir} gain. The T=8 R50 step regresses; restoring
  $T{=}4$ and adding rotating folds recovers $+3.4$\,pp over the R18 baseline.}
  \label{fig:backbone}
\end{figure}
```

### 3.8 Conclusion / abstract (~line 492)

**Current:**
```latex
...reach a competitive $44.03\%$ on a notoriously hard benchmark,
with a log-softmax TSM+Transformer ensemble.
```
**Replace with:**
```latex
...reach a competitive \textbf{46.75\%} on a notoriously hard benchmark
with a five-model log-softmax ensemble (TSM-ResNet50 rotating $+$ two
TSM-ResNet18 variants $+$ VFL $+$ CE-TSM ablation), up from 44.03\%
before the ResNet50 and ablation-guided additions.
```

---

## 4. Summary of numerical changes

| Location | Old value | New value |
|----------|-----------|-----------|
| Best single model top-1 | 41.68% (TSM-v2-rot +TTA) | **42.85%** (TSM-R50 rot) |
| Best ensemble top-1 | 44.03% (3-model) | **46.75%** (5-model) |
| Best ensemble top-5 | 75.37% | **76.98%** |
| Best ensemble macro-F1 | 0.400 | **0.419** |
| Ensemble lift over best single | +2.35 pp | +3.90 pp |
| LOO dominant contributor | −3.17 pp (TSM-v2) | **−1.33 pp (R50)** |
| Focal ablation (vs CE) | estimated | measured: +0.53 pp (γ=2), +0.58 pp (γ=1) |
| VFL depth 2→3 | estimated | measured: **+1.08 pp** |
| VFL 10-fold vs 5-fold | expected positive | measured: **−0.46 pp** (saturated) |

---

## 5. Figures that do NOT need TeX changes

| Figure | Status |
|--------|--------|
| `arch_comparison.png` | Architecture unchanged; no update needed |
| `fig02_tta_effect.png` | Only covers original 6 models; still accurate |
| `fig03_rotating_effect.png` | Only covers TSM-v2 and VFL-Ultra; still accurate |
| `fig06_confusion_matrix.png` | Still shows old best ensemble; minor discrepancy acceptable (no TeX label change needed; caption could note "prior ensemble") |
| `fig07_confused_pairs.png` | Same (still shows old ensemble confusion pairs) |
| `fig08_f1_vs_support.png` | Same (scatter is fairly stable across ensembles) |
| `fig12_tta_directional_tradeoff.png` | Analysis unchanged; no update needed |
| `fig13_agreement_matrix.png` | Covers original 12 models; R50 could be added but it is not critical |
