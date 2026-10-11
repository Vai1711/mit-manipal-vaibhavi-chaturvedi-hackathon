# Jury questions and defensible answers

**Why this model?** The trained TF-IDF/logistic scorer improves on a word list while remaining very small. On the packaged 240-row sentiment sample it scored 76.3%, versus 73.3% for the actual DistilRoBERTa reference. This does not establish universal superiority. Event classification uses ModernBERT semantic inference with evidence-rule anchors and review gates.

**Did you train on your evaluation data?** No. Normalized duplicate/conflicting original training texts were handled before a stratified 80/20 development split. C was chosen among 0.3, 1 and 3 using development macro F1. Vocabulary and weights use training only; no development refit. Validation texts with normalized exact training overlap are excluded. This cannot exclude semantic duplicates or pretrained model exposure.

**Are the event labels independent?** No. They are builder-curated, keyword-enriched examples selected from a public topic validation split. The six labels are not the original publisher topic taxonomy. The 90.7% result is a reproducible diagnostic, not a representative deployment estimate. Next step: blinded independent annotation and out-of-period evaluation.

**Why not force a Fed hike to negative sentiment?** Text tone and exposure direction are different. A neutral rate announcement can have hawkish stance and a signed rate-rise scenario. That scenario affects bonds and swaps through their sensitivities; sentiment alone does not determine the shock.

**Is the default engine AI?** Its sentiment component is a genuinely trained model running on any text. Packaged event texts use actual recorded semantic predictions with model and prompt provenance. New uncached text uses disclosed evidence rules: explicit supported actions can pass existing safety gates, while ambiguous reports require review; optional local AI computes semantic predictions for new inputs. Recorded replay is never claimed to be new live inference.

**Why confidence 0.65, margin 0.10 and severity >7?** They are explicit conservative design choices. They have not been optimized or calibrated on independent outcomes. Model scores are relative category scores, not correctness probabilities; review also considers assertion, currency, direction and source context.

**What does 1–10 impact mean?** A transparent scenario-priority rubric combining event base severity, language cues, breadth and uncertainty. It is not trained market-impact prediction. Six SPY event-day observations illustrate a validation route but cannot establish causation, calibrate severity or evaluate borrower-specific loss.

**How do you calculate market P&L?** For a bond, ΔPV ≈ −PV × modified duration × Δyield (bps/10,000). The pay-fixed swap uses a signed rate sensitivity and gains under rate rises; it loses under rate cuts. There is no full curve repricing, convexity, basis or counterparty valuation adjustment.

**How do you calculate credit loss?** Incremental ECL = EAD × (stressed PD − baseline PD) × LGD, with PD capped at 1. This is a one-year simplified credit calculation, not lifetime IFRS 9 provisioning. Loan EAD is not market value. Swap notional is not swap PV. The portfolio-value chart excludes loan EAD and hedge notional; market P&L and ECL remain separate.

**Does RSS satisfy real-time?** It demonstrates explicit periodic ingestion of public titles with original timestamps. It is not continuous streaming or guaranteed fresh. News and social adapters have independent visible snapshot fallback; uncertain/unmapped items require review. No polling or downloads happen by default.

**Why only Module B?** The brief requires at least one downstream module. Concentrating on exposure mapping, hedge math and traceability allows stronger validation than spreading effort across two shallow modules.

**What is still missing before submission?** Candidate details, public named GitHub repository and actual unlisted demo recording. The bundle supplies code, data, tests, architecture and a seven-slide PDF/PPTX. AI assistance should be disclosed, and the candidate must understand and verify the work.
