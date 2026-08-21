# STATE — PINN-for-composite-interface-identification

<!-- Machine-maintained by save-session Step 6b. Do not hand-edit. -->

Status: shipped
Last touched: 2026-08-21

## What
Master's thesis payload (defended 12-Mar-2026): PINN bridging 3-Layer Interphase and Extended Interface models in composite homogenization — finds interface parameters reproducing K_eff/G_eff. PyTorch. PUBLIC repo on GitHub.

## Done
- Thesis defended; repo public with MIT license
- README metrics scoped to 28 held-out validation configs, 100% claims softened (2026-05-27, per no-cherry-picked-metrics rule) — SUPERSEDED, see below
- README links Master-Thesis LaTeX source repo (2026-05-31)
- README fully realigned to thesis Ch. 4 (2026-08-21, commit 3e431ac). The 28-config set existed in NO report — pre-submission protocol, dropped. Replaced with the four evaluations the thesis actually runs: both worked-example tables, 500 random configs (K 82% / G 56% pass, median 1.4% / 4.0%), 33/36 sweep. Added EGIM terminology, architecture + loss spec, CCA structural failure, real citation
- PROJECT_SUMMARY.md in the thesis workspace re-based on the same numbers (2026-08-21, not a git repo)

## Doing
- Nothing in progress

## Pipeline
- Occasional future updates only (user: "maybe I will update some stuff in future")

## Resume here
No planned work. Commit 3e431ac is local — push line was printed at session end. Two open
questions if returning: (1) thesis says 100k training samples, checkpoints/v2/training_config.json
says 50k, both best_epoch 322 — unresolved; (2) does the two-step Mori-Tanaka bug (found
2026-02-24, too late to fix) explain why G_eff (56% pass) trails K_eff (82%)? If yes,
retraining on corrected targets is the highest-value move. If touching README: re-read
no-cherry-picked-metrics rule first.

## Landmines
- PUBLIC repo — full git history visible; no unverifiable claims, no PII
- Results language deliberately softened — do not re-inflate ("100% pass" framing was killed on purpose)
- README numbers must trace to thesis Ch. 4 tables, NOT the abstract — the abstract's "errors below 2%" is contradicted by the thesis's own Table 1 (case 3 = 10.68% G_eff). main.tex:3047 also carries an orphan "100% pass rate" claim. Thesis left unfixed on purpose (defended document)
- Do NOT port files from Thesis_Final/Thesis_Project/ into this repo. The 500-sample eval script lives only there; that was accepted rather than fixed by copying
- Repo harness not initialized (no docs/exec-plans/, BACKLOG.md, SESSION-END.md, Archive/). Project is shipped — probably fine, but noted
