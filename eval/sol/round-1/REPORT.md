# Round 1: independent AI-detector review

## Verdict

**Our AI-detection quality score: 9.0/10. ZeroGPT reference score on exactly the same texts: 10.0/10.** These are the diagnostic scores specified before the full results were inspected, not universal quality ratings.

**Is ours at least as good as ZeroGPT on this set? no.**

Ours is useful, but the comparison is not close enough to claim parity: at the requested 50% detector thresholds, ours makes 12 errors and ZeroGPT makes one. Excluding the inherently ambiguous AI-polished-human category, ours makes ten errors and ZeroGPT makes none. ZeroGPT has both fewer human false positives and better AI sensitivity; there is no compensating category-level advantage for ours. The stricter local `/verify` gate catches two more AI-derived texts but wrongly blocks four additional humans, so it does not close the gap.

## Method and integrity checks

This report analyzes the completed, frozen evaluation; **neither detector was rerun, no text was changed, and no detector code or service was changed**. I read `BRIEF.md`, the pre-outcome `ANALYSIS_PLAN.md`, `analyze.py`, the saved summary and completion log, and independently recomputed confusion counts and category flag counts from `results.json`. I also checked every result's SHA-256 against the actual corresponding text and its stored hash. All 120 unique text IDs matched. The whole `texts.json` file still matches the evaluation's frozen manifest hash:

`fa465126e7b4e635f0cda33440424e24e93ecd74ae41cbc096aaa7229ca58221`

The set was frozen at 2026-11-11 21:44:18 UTC; the evaluation ran from 21:45:35 to 22:17:51 UTC. `evaluation.log` ends with `EVALUATION COMPLETE`. All three endpoints have successful results for **120/120** texts, with no missing/error predictions. Every saved verify response reports the same versions: engine `tropa-mini-ai-text-detector@10fecd08+hybrid`, service `1`, tells `2`. The `/api/detect` score agrees with the score in `/verify` on every text. All local detections ran on CPU. The saved outputs therefore evaluate the **hybrid version actually returned by the service**, not an assumed earlier implementation.

### Dataset

There are **120 texts, 66–388 words**, comprising 60 human texts, 50 generated AI texts, and 10 AI-polished human texts:

- **30 documented non-native human texts:** original Cambridge FCE exam scripts from 2000–2001, published in 2011 and distributed in BEA2019 v2.1, with recorded non-English first languages. These cover letters, complaints, and other exam genres.
- **30 native/proficient-human proxies:** 8 pre-2022 Workplace Stack Exchange posts, 4 pre-2022 Academia Stack Exchange posts, 8 historical business-letter examples from Mary Owens Crowther's *How to Write Letters*, and 10 Brown Corpus excerpts from 1961. The label does **not** establish the individual first language of every writer. Historical model letters are human-written examples, not genuine modern applicant records.
- **20 raw AI texts** and **30 casual/humanized AI texts:** generated before outcome inspection. The hard group includes 8 casual texts, 8 humanize/paraphrase rewrites, 7 all-lowercase texts, and 7 non-native-English imitations.
- **10 AI-polished human texts:** public human originals rewritten by an AI, retained as a separate category rather than silently equated with wholly AI-authored writing.

Across the 60 AI-derived texts, recorded model identifiers are `claude-haiku-5-5` (18), `gemini-3-flash` (18), `gpt-oss-120b-medium` (18), and reviewer-written `gpt-6.1-sol` (6). Identifiers are recorded provenance, not independently authenticated model identities. The set covers emails, follow-ups, reference requests, cover letters, application answers, LinkedIn-style posts/messages, essays, reviews, forums, and fiction. Source URLs, document identifiers, generation specifications, and derivative links are recorded in `texts.json`. Human sources are outside the brief's listed previously used collections; this does not prove absence from detector pretraining.

### Decisions and scoring fixed in advance

- **Ours:** flagged AI if `/api/detect` `ai_score >= 0.50`.
- **ZeroGPT:** flagged AI if `fakePercentage >= 50`.
- **Local agent gate:** separately report `/verify` FAIL, using its default 30% maximum and strong-tell rules. FAIL is a block, not an independently calibrated AI probability.
- **Overall accuracy:** report both all 120 texts, treating AI-polished texts as AI-derived positives, and the 110-text subset excluding polishing.
- **Quality score:** `10 × (0.40 × human specificity + 0.20 × raw-AI sensitivity + 0.40 × casual/humanized-AI sensitivity)`. Polishing is excluded from this score because its target label is debatable. Identical weights are applied to ZeroGPT.
- The predeclared parity rule requires at least ZeroGPT's non-polished accuracy, no higher aggregate human false-positive rate, and no lower hard-AI sensitivity. Ours fails all three conditions.

## Per-category results

All entries are **flagged-as-AI counts and rates**. For human rows, lower is better: these are false positives. For AI rows, higher is better. The gate column gives FAIL counts, not predictions at the 50% threshold.

| Category | N | Ours: AI score ≥50% | ZeroGPT: AI score ≥50% | Ours `/verify`: FAIL |
|---|---:|---:|---:|---:|
| Human native/proficient proxy | 30 | 2/30 — **6.7%** | 0/30 — **0.0%** | 2/30 — **6.7%** |
| Human non-native | 30 | 3/30 — **10.0%** | 0/30 — **0.0%** | 7/30 — **23.3%** |
| AI raw | 20 | 19/20 — **95.0%** | 20/20 — **100.0%** | 19/20 — **95.0%** |
| AI casual or humanized | 30 | 26/30 — **86.7%** | 30/30 — **100.0%** | 28/30 — **93.3%** |
| AI-polished human | 10 | 8/10 — **80.0%** | 9/10 — **90.0%** | 8/10 — **80.0%** |

Within the hard group, both detectors catch all 8 casual and all 8 humanize/paraphrase examples. Ours catches **6/7 lowercase** and only **4/7 non-native imitations**, versus ZeroGPT's 7/7 in each. `/verify` catches 6/7 in each of these latter subgroups. The observed hard-case gap is therefore concentrated in lowercase and non-native imitation, not every kind of humanization.

### Overall accuracy and confusion counts

TP = AI-derived text flagged; TN = human text not flagged; FP = human text flagged; FN = AI-derived text missed.

| Evaluation | Detector/gate | Correct / N | Accuracy | Balanced accuracy | TP | TN | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| All texts, polishing counted positive | Ours | 108/120 | **90.0%** | 90.0% | 53 | 55 | 5 | 7 |
| All texts, polishing counted positive | ZeroGPT | 119/120 | **99.2%** | 99.2% | 59 | 60 | 0 | 1 |
| All texts, polishing counted positive | Ours `/verify` | 106/120 | **88.3%** | 88.3% | 55 | 51 | 9 | 5 |
| Excluding AI-polished human | Ours | 100/110 | **90.9%** | 90.8% | 45 | 55 | 5 | 5 |
| Excluding AI-polished human | ZeroGPT | 110/110 | **100.0%** | 100.0% | 50 | 60 | 0 | 0 |
| Excluding AI-polished human | Ours `/verify` | 98/110 | **89.1%** | 89.5% | 47 | 51 | 9 | 3 |

Across the 60 humans, the detector's false-positive rate is **8.3%** and the gate's wrongful-block rate is **15.0%**, versus ZeroGPT's **0.0%**. Counting all AI-derived texts, sensitivity is 88.3% for ours and 98.3% for ZeroGPT.

On the 110-text non-polished subset, there are **10 ZeroGPT-only correct decisions and zero ours-only correct decisions**, a 9.1-percentage-point accuracy gap. Exact two-sided McNemar p = **0.001953125** under independent-pair assumptions. Sources, exam prompts, and derivative examples cluster, so this is descriptive evidence for this set, not a population-level guarantee. Category uncertainty is substantial: approximate 95% Wilson intervals are 1.8–21.3% for ours' native/proficient false-positive rate, 3.5–25.6% for its non-native rate, and 70.3–94.7% for its hard-AI sensitivity. ZeroGPT's observed 0/30 human errors still has a Wilson upper bound of 11.4%; 30/30 hard-AI detections has a lower bound of 88.6%.

## Ten instructive local failures

Scores below are the saved detector outputs expressed as percentages, not probabilities proven to be calibrated. Related examples are grouped to avoid treating the same failure pattern as independent discoveries.

| Case and known provenance | Short quote | Ours | ZeroGPT | `/verify` | Lesson |
|---|---|---:|---:|---|---|
| **H005**, human FCE complaint, Turkish L1; related H009 Polish L1 and H003 Thai L1 | “I am writing to inform you about my disappointing evening” | 99.99% | 0% | FAIL | All three genuine 2000-era complaints are effectively declared certain AI. Their model `humanized` class probabilities are 99.99%, 99.96%, and 99.90%, despite near-zero `ai` class probability. Similar task templates cluster these errors. |
| **H048**, historical human reference request | “a recommendation from you would help greatly” | 60.27% | 0% | FAIL | Ordinary formal job-search language is a false positive in a directly relevant genre. Removing this sentence changes the local score to 1.49%, illustrating instability rather than proof that the sentence is machine-authored. |
| **H058**, Brown Corpus human fiction, 1961 | “It was the first time any of us had laughed since the morning began.” | 99.40% | 0% | FAIL | `humanized` probability is 99.36%, while `human` is only 0.60%. Several single-sentence removals collapse the reported AI score to nearly zero. The output is highly sensitive to crossing the apparent score-aggregation regime. |
| **GH02**, Gemini lowercase application answer | “i actually use the tool every single day for managing my personal side projects” | 12.06% | 100% | PASS | A plausible lowercase application answer passes both local endpoints despite being wholly generated. This is a core use case, not just a fiction edge case. |
| **S004**, reviewer-generated non-native-style reference request | “they ask me to give one teacher reference because my working experience is not much yet” | 9.43% | 100% | PASS | `humanized` probability is 62.67%, yet the reported AI score uses a much smaller value. The model has relevant signal that is not reflected in the binary decision. |
| **S002**, reviewer-generated raw LinkedIn-style post | “The useful part was not choosing a new platform.” | 2.50% | 100% | PASS | Even untransformed AI is not completely covered: a concise, concrete workplace narrative evades both the model threshold and the tell gate. |
| **GH03**, Gemini non-native-style cover letter | “I am writing for apply to the receptionist position” | 15.24% | 100% | FAIL | The detector misses it despite `humanized` probability 56.06%; the gate catches it only through “Thank you for your time and consideration.” This is not a robust recovery of authorship signal. |
| **KP04**, Haiku-polished human book review, parent H051 | “he has achieved a notable failure” | 0.24% | 61.3% | PASS | `ai_edited` probability is 93.06%. The raw class output correctly identifies editing, but the single AI score and gate hide that distinction. This is a miss only under the explicitly stated AI-derived-positive convention. |
| **GP03**, Gemini-polished human fiction, parent H057 | “She was becoming overly dramatic.” | 5.34% | 0% | PASS | Both detectors miss this polishing operation under the AI-derived convention. Ours has `humanized` probability 91.73%, again showing the tension between class outputs and the collapsed score. Do not describe this as proof of fully generated authorship. |
| **H015**, human FCE letter; same gate-only failure in H007, H020, H012 | “I look forward to hearing from you” | 0.02% | 0% | FAIL | A strong stock-closer rule vetoes an otherwise correct human decision. All four additional human blocks are this rule, with detector scores between 0% and 0.03%. This closing demonstrably predates LLMs. |

The remaining hard-AI threshold miss, **OH03**, scores 42.74% locally and 100% at ZeroGPT; `/verify` catches it using the stricter 30% threshold. Thus the two extra AI detections obtained by the gate come from different mechanisms: OH03 from the threshold and GH03 from a tell. Its four additional false positives all come from the stock-closer rule, not borderline model scores.

The leave-one-out highlights deserve particular caution. For H005, some reported drivers change a score of about 100% by only hundredths of a percentage point, while other examples change nearly discontinuously. A driver is a counterfactual model sensitivity, not evidence that a human sentence should be rewritten or that it was AI-authored.

## Score and rationale

Applying the frozen scorecard gives:

- Ours: `10 × (0.40 × 55/60 + 0.20 × 19/20 + 0.40 × 26/30) = 9.0333`, rounded to **9.0/10**.
- ZeroGPT: `10 × (0.40 × 60/60 + 0.20 × 20/20 + 0.40 × 30/30) = 10.0`, or **10.0/10**.

The 9.0 score reflects genuinely strong basic discrimination: ours preserves 55/60 humans, catches 19/20 raw AI texts, and catches 26/30 hard AI texts. It does **not** imply equality with the reference. On the same weighted components, ZeroGPT is perfect in this small sample and ours is behind in every component. The most consequential weaknesses are confidently mislabeling genuine non-native letters, missing generated text that imitates imperfect English, and allowing a conventional closer to override a nearly certain human model decision. Polishing remains a separate, ambiguous target and does not influence the score. The score is a predefined weighted diagnostic; moving its scale after seeing outcomes would be inappropriate. The gate happens to receive the same weighted diagnostic score because its extra hard-AI catches offset its lost human specificity under these weights, but its overall accuracy is lower.

## What would most improve ours

### 1. Replace the discontinuous class-to-score rule with held-out calibration

The stored class vectors expose a specific problem: several real human texts are assigned nearly 100% `humanized` and then reported as nearly 100% AI, while generated non-native-style texts with 56–63% `humanized` are reported at only 9–15% AI. The observed scores are consistent with switching from `P(ai)` to `1 − P(human)` only when `P(human)` becomes very small. Audit and replace this abrupt aggregation with a calibrated, continuous combination of class outputs, learned and validated on **other** data with native/proficient and non-native humans represented. Assess subgroup false positives and raw/hard-AI recall jointly. Do not simply use `1 − P(human)` everywhere: that can exchange one bias for another, and the class model itself is overconfident on the FCE false positives. Expose uncertainty and distinguish AI-generated, humanized, and AI-edited outputs rather than presenting one score as certainty.

### 2. Improve robust detection of non-native imitation and lowercase workplace AI

Prioritize diverse, independently generated emails, application answers, reference requests, and cover letters with imperfect grammar, lowercase formatting, concrete details, and restrained prose, alongside matched genuine pre-2022 human examples. Here, non-native imitation is only **4/7** caught at 50%, and two directly relevant generated texts pass even the gate. Validate by held-out source, prompt family, and model, not random text splits that leak related examples. A pure threshold reduction will not solve both confidently wrong human decisions and low-scoring AI misses. Develop on separate material; keep this round frozen as an audit set, and use a new unseen round for final claims after changes.

### 3. Stop treating ordinary human closers as conclusive AI evidence

Demote conventional phrases such as “I look forward to hearing from you” to optional style advice unless independent evidence supports a block. Separate an authorship verdict from a writing-style checklist and make the reason for any block explicit. On these saved outputs, removing the stock-closer veto **alone** would rescue four genuine humans but also release GH03, one generated cover letter: the gate would move from 106/120 to **109/120 (90.8%)** correct. This is a retrospective arithmetic counterfactual, not a newly tested service result, and remains far below ZeroGPT's 119/120. The goal is a better-evidenced gate, not disabling rules indiscriminately.

Additional reporting improvement: expose the 93.06% `ai_edited` signal in KP04 separately, with an explicit policy for whether polishing is allowed. A generic AI percentage currently conceals information useful to the owner.

## Limits on interpretation

- This is an intentionally challenging, source-clustered set, not a random sample of the owner's future writing. Small category counts and related prompt/derivative families limit precision.
- The native/proficient stratum includes unknown individual first languages. All documented non-native examples come from FCE; three detector false positives share a theatre-complaint task. Neither group estimates a universal language-background error rate.
- Historical business letters and exam letters are proxies. There are no authentic pre-2022 LinkedIn posts or modern application-form answers here, and short texts dominate. The maximum observed length is 388 words, so no conclusion is supported about longer documents.
- Public source novelty relative to the builder's exclusion list does not establish that sources were absent from training. Historical examples may be familiar to either detector.
- AI-polished human text has mixed provenance. Calling it an error depends on an AI-derived policy; that is why both the full and polishing-excluded results are reported.
- A live public detector can change. This conclusion is about these frozen answers from 2026-11-11, not guaranteed future ZeroGPT behavior. Its 10.0 reference score does not establish universal perfection.
- No private user writing was needed for this comparison. No new external submissions were made while finishing this report.

**Bottom line: ours is not at least as good as ZeroGPT on this set.** Improve calibrated treatment of the `humanized` class, hard workplace AI, and the gate's overbroad phrase vetoes before claiming otherwise.
