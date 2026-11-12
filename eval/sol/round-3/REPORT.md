# Round 3 — independent detector review, phase B

## Bottom line

**Primary, frozen round-3 quality score: ours 9.03/10; ZeroGPT 9.80/10.**

**Round-2-set rerun: ours 9.73/10 (unrounded 9.733333); saved ZeroGPT 9.80/10.** The same local score was 9.40/10 at round 2. This is a matched-set improvement, but **v7 training templates were partly informed by round 2's reported failures: round 2 is no longer a clean unseen test**. Treat it as a development-informed regression check. **Round 3 is primary.** Different-set scores must not be interpreted as a measured engine regression on round 3; there is no previous-engine round-3 run here.

**Is ours at least as good as ZeroGPT on this set? no.**

**Is this a frontier-level AI-text detector? no.**

The decisive issue is out-of-sample separation of ordinary, restrained professional AI from short genuine human writing. On round 3, ours misses 7/57 generated priority messages versus ZeroGPT's 2/57, while falsely flagging 1/23 human priority proxies versus 0/23. The stricter `/verify` gate still passes 6/57 generated priority messages and wrongly blocks 3/23 humans. On 30–60-word texts, ours catches only 16/21 wholly generated texts and flags 2/22 humans. Those outcomes do not support parity or a frontier claim, irrespective of the high fixed aggregate score.

**Overall judgment:** a useful local screening aid, not yet a dependable automatic release gate for the owner's casual-professional job-search writing. The fixed 9.03 is the prespecified weighted diagnostic, not a calibrated probability of correctness or a safety rating. Formal human specificity and application-answer detection are encouraging; strong gains on development-informed examples are not sufficient evidence of generalization.

**Single biggest improvement needed:** improve the separation of **short, natural, specific professional messages**, using independently sourced, genre/length-matched genuine human and generated examples, with entire sources, authors, prompt families and generator families held out. Optimize sensitivity subject to a low human false-positive ceiling; consider an explicit uncertain/abstain outcome for short ambiguous text. Do not fix this by globally lowering thresholds or adding boilerplate vetoes: several missed AI messages score 0.5–14%, while genuine humans score 54–86%. Use other development data, not these audit texts.

## Method, immutable inputs, and actual deployed engine

Read-only frozen round-3 `texts.json` was verified before any phase-B model request and again after evaluation. Its SHA-256 is unchanged:

`20efbed370ffba61f7434286652e8ba5ffdf4e72f52af12767235e349c27abfd`

It was frozen at **2026-11-12T08:25:23.560612+00:00**, before phase A's first detector request. All 160 individual text hashes match. Round 2's unchanged file hash is:

`45f4711b900d4f2ddf4ee473aa5459edf10642be7aa4469b4e595d46e00ab247`

The actual `/healthz` engine string before and after both runs was:

`rp-modernbert-v6@1579161232-1794457937 + rp-modernbert-v7@1579161232-1794473864+stack@0d0c5cda`

All **280 `/verify` responses** report that same engine, `service: 1`, `tells: 3`, `complete: true`, and `from_store: false`. All 280 `/api/detect` responses report `device: cuda`. `/api/detect` and `/verify` AI scores agree on every text. Before/after raw health responses, per-attempt timestamps, raw HTTP response bodies and parsed responses are retained in the phase-B result files.

| Run | UTC start | UTC finish | `/api/detect` | `/verify` | Local retries |
|---|---|---|---:|---:|---:|
| Round 3, primary | 2026-11-12 09:06:25 | 2026-11-12 09:10:11 | 160/160 | 160/160 | 0 |
| Round 2, regression check | 2026-11-12 09:10:11 | 2026-11-12 09:15:10 | 120/120 | 120/120 | 0 |

The logged background job called `/api/detect`, then `/verify`, one request at a time. Audit of all 560 attempt intervals confirms no overlap. There were no incomplete responses and no missing predictions silently counted as human.

**No new ZeroGPT requests were made.** Round 3 reuses all 160 answers in `zerogpt.json`; round 2 reuses all 120 answers in its original `results.json`. Each reused result's ID/hash and raw result object were checked against its original saved file. The original round-2 files were not overwritten. No detector code, models, or service configuration was edited, no service was restarted, no generation model was called in phase B, and nothing was committed. No Fable calls were made.

### Fixed scoring rules

- Local detector flags AI at `ai_score >= 0.50`.
- ZeroGPT flags AI at `data.fakePercentage >= 50`.
- `/verify` FAIL is reported separately. All tested responses have `max_ai: 0.30`; no strong-tell veto occurred in either run. Every observed gate verdict agrees with the reported 30% threshold. Soft tells are not independent rejection reasons.
- Unchanged fixed formula for both detectors:

  `10 × (0.40 × human specificity + 0.20 × raw-AI sensitivity + 0.40 × casual/humanized-AI sensitivity)`

- AI-polished human is mixed provenance and excluded from that score and primary accuracy. All-text accuracy additionally counts it as AI-derived positive, strictly as an alternate labeling convention, not a claim that light editing erases human authorship.
- Prespecified parity requires at least ZeroGPT's polishing-excluded accuracy, no higher aggregate human FPR, and no lower casual/humanized-AI sensitivity. Ours fails all three on round 3.

Score arithmetic:

- Round 3, ours: `10 × (0.40 × 77/80 + 0.20 × 21/25 + 0.40 × 35/40) = 9.03`.
- Round 3, ZeroGPT: `10 × (0.40 × 80/80 + 0.20 × 25/25 + 0.40 × 38/40) = 9.80`.
- Round 2 rerun, ours: `10 × (0.40 × 58/60 + 0.20 × 20/20 + 0.40 × 29/30) = 9.733333`.

`analyze_phase_b.py` validates hashes, saved ZeroGPT identity, completion, response bodies, versions and serial requests, and computes the tables. A separate direct recomputation of the score components agrees; see `phase_b_audit.json`.

## Dataset and provenance

Round 3 has **160 texts, 32–396 words**: 80 human, 65 wholly generated AI, 15 AI-polished human. **92** are in priority genres and **48** are 30–60 words inclusive. No exact/normalized text overlap exists with rounds 1–2.

Human sources: **40 PELIC v1.1** learner texts collected 2005–2012 (40 distinct anonymous authors, recorded L1s); **12 Stevenson letters** published in 1899; **6 Huxley scientific/philosophical excerpts** from *Darwiniana* (1893); **6 Wells narrative excerpts** from *The Time Machine* (1895); **8 Bicycles and 8 Home Improvement Stack Exchange questions**, created before 2018 with no post-2021 edits in the acquired metadata. All human texts are pre-2022. Source URLs, IDs, dates, original hashes and excerpt metadata are in `texts.json`; full sourcing and generation decisions are in `PHASE_A.md`. Forum authors' L1s are unknown: “native/proficient” is a proxy, not verified native-speaker status.

AI sources: **20 texts each** from requested proxy models `gpt-6.1-terra`, `gemini-3.1-flash-lite`, `claude-sonnet-4-6`, and reviewer writing labeled `gpt-6.1-sol`. Requested/harness labels are provenance, not independent authentication of underlying model identities. Proxy prompts and raw generations are retained under `generation/`. All selections and documented source/placeholder corrections preceded freeze and all detector outcomes; no frozen text was changed in phase B.

**Human priority coverage remains limited:** the 23 human priority texts are classroom application answers/thanks and historical professional correspondence. There are **no verified authentic modern human LinkedIn messages/posts or recruiter replies**. Human and AI priority sides are not fully genre/era matched. This is a limitation of the audit, not permission to claim modern-human specificity from the aggregate.

## Round 3 — per-category results

Flags are false positives for human rows, sensitivity for wholly generated AI, and separately reported AI-derived flags for polishing. Gate counts are FAIL rates.

| Category | N | Ours ≥50% | ZeroGPT ≥50% | Ours `/verify` FAIL |
|---|---:|---:|---:|---:|
| Human native/proficient proxy | 40 | 2/40 — 5.0% | 0/40 — 0.0% | 3/40 — 7.5% |
| Human non-native | 40 | 1/40 — 2.5% | 0/40 — 0.0% | 2/40 — 5.0% |
| AI raw | 25 | 21/25 — 84.0% | 25/25 — 100.0% | 22/25 — 88.0% |
| AI casual/humanized | 40 | 35/40 — 87.5% | 38/40 — 95.0% | 36/40 — 90.0% |
| AI-polished human | 15 | 8/15 — 53.3% | 7/15 — 46.7% | 8/15 — 53.3% |

| Accuracy convention | Ours | ZeroGPT | Ours `/verify` |
|---|---:|---:|---:|
| Primary: polishing excluded | 133/145 — 91.72% | 143/145 — 98.62% | 133/145 — 91.72% |
| All texts, polishing counted AI-derived positive | 141/160 — 88.12% | 150/160 — 93.75% | 141/160 — 88.12% |

Primary confusion counts: ours **TP 56, TN 77, FP 3, FN 9**; ZeroGPT **TP 63, TN 80, FP 0, FN 2**; gate **TP 58, TN 75, FP 5, FN 7**. Of primary paired decisions, ZeroGPT alone is correct on 10 texts and ours alone on none. Ours' one-extra polishing flag does not compensate for this primary gap, and the polishing labels themselves are ambiguous.

The gate rescues two generated-AI misses at 50% (`R3M1R05`, scientific prose, 45.92%; `R3M1H02`, follow-up, 43.78%) but creates two additional human rejections (`R3H009`, 36.60%; `R3H042`, 35.32%). It therefore leaves primary accuracy unchanged, with a different and more intrusive error trade-off. `R3H009`'s saved leave-one-out explanation attributes the shift from 36.60% to 6.45% to omitting “Sincerely,”; this is the service's attribution output, not an additional modified-text experiment or a proven causal rule. Its “not only … but also” tell is soft, not the veto.

## Priority-use slice — primary round 3

| Priority group | N | Ours ≥50% | ZeroGPT ≥50% | `/verify` FAIL |
|---|---:|---:|---:|---:|
| Human proxies — false positives | 23 | 1/23 — 4.3% | 0/23 — 0.0% | 3/23 — 13.0% |
| Wholly generated AI — sensitivity | 57 | 50/57 — 87.7% | 55/57 — 96.5% | 51/57 — 89.5% |
| Raw AI subset | 17 | 15/17 — 88.2% | 17/17 — 100.0% | 15/17 — 88.2% |
| Casual/humanized AI subset | 40 | 35/40 — 87.5% | 38/40 — 95.0% | 36/40 — 90.0% |
| AI-polished human — separate | 12 | 8/12 — 66.7% | 5/12 — 41.7% | 8/12 — 66.7% |

Priority accuracy excluding polishing: **ours 72/80 (90.0%), ZeroGPT 78/80 (97.5%), gate 71/80 (88.8%)**. Every round-3 hard-style example is in a priority genre; this is not an essay-heavy proxy for the requested use case.

### Per-priority-genre generated-AI sensitivity

| Genre | N | Ours | ZeroGPT | Gate |
|---|---:|---:|---:|---:|
| LinkedIn outreach | 7 | 6/7 — 85.7% | 6/7 — 85.7% | 6/7 — 85.7% |
| LinkedIn posts | 8 | 8/8 — 100% | 8/8 — 100% | 8/8 — 100% |
| Recruiter replies | 8 | 6/8 — 75.0% | 7/8 — 87.5% | 6/8 — 75.0% |
| Networking/referral asks | 8 | 6/8 — 75.0% | 8/8 — 100% | 6/8 — 75.0% |
| Follow-ups | 8 | 6/8 — 75.0% | 8/8 — 100% | 7/8 — 87.5% |
| Thank-you notes | 6 | 6/6 — 100% | 6/6 — 100% | 6/6 — 100% |
| Application answers | 12 | 12/12 — 100% | 12/12 — 100% | 12/12 — 100% |

Human-side priority coverage and false positives: application answers **0/7 ours, 0/7 ZeroGPT, 0/7 gate**; historical follow-ups **0/4, 0/4, 0/4**; historical networking/referral asks **0/2, 0/2, 1/2**; thank-you notes **1/10, 0/10, 2/10**. There are no human LinkedIn/recruiter examples; an absent subgroup is not a 0% FPR result.

The 12 priority polishing samples comprise 3 application answers, 5 thanks, 2 referral asks and 2 follow-ups. Their flags are kept separate in `phase_b_tables.md` and `phase_b_summary.json`, never folded into wholly generated sensitivity.

## 30–60-word slice — primary round 3

| Group | N | Ours ≥50% | ZeroGPT ≥50% | `/verify` FAIL |
|---|---:|---:|---:|---:|
| Human — false positives | 22 | 2/22 — 9.1% | 0/22 — 0.0% | 2/22 — 9.1% |
| Wholly generated AI — sensitivity | 21 | 16/21 — 76.2% | 19/21 — 90.5% | 17/21 — 81.0% |
| Raw AI subset | 7 | 6/7 — 85.7% | 7/7 — 100% | 6/7 — 85.7% |
| Casual/humanized AI subset | 14 | 10/14 — 71.4% | 12/14 — 85.7% | 11/14 — 78.6% |
| AI-polished human — separate | 5 | 1/5 — 20.0% | 2/5 — 40.0% | 1/5 — 20.0% |

Short-slice accuracy excluding polishing: **ours 36/43 (83.7%), ZeroGPT 41/43 (95.3%), gate 37/43 (86.0%)**. Including polishing under the alternate positive convention: **37/48 (77.1%), 43/48 (89.6%), 38/48 (79.2%)**, respectively. There are no 30–60-word texts in round 2; no short-slice change can be estimated there.

The two short human false positives are a 35-word PELIC learner reflection and a 57-word Stevenson thank-you. A short text or a naturally imperfect sentence must not be treated as sufficient evidence of AI authorship. These small, source-clustered samples are diagnostics, not population estimates.

## Human bias table — round 3

All entries below are false-positive rates, except that the gate column is wrongful FAIL. Groups overlap and must not be summed. The requested inclusive 30–60-word group is primary for short-text reporting; the ≤150/>150 split is retained as a secondary comparison with earlier reporting. Formality is the frozen annotation; it is not inferred from L1 or detector scores.

| Human subgroup | N | Ours FP | ZeroGPT FP | Gate wrongful FAIL |
|---|---:|---:|---:|---:|
| Native/proficient proxy | 40 | 2/40 — 5.0% | 0/40 — 0.0% | 3/40 — 7.5% |
| Non-native | 40 | 1/40 — 2.5% | 0/40 — 0.0% | 2/40 — 5.0% |
| Formal | 31 | 0/31 — 0.0% | 0/31 — 0.0% | 0/31 — 0.0% |
| Casual | 32 | 2/32 — 6.3% | 0/32 — 0.0% | 2/32 — 6.3% |
| Casual-professional | 17 | 1/17 — 5.9% | 0/17 — 0.0% | 3/17 — 17.6% |
| 30–60 words | 22 | 2/22 — 9.1% | 0/22 — 0.0% | 2/22 — 9.1% |
| More than 60 words | 58 | 1/58 — 1.7% | 0/58 — 0.0% | 3/58 — 5.2% |
| ≤150 words | 54 | 3/54 — 5.6% | 0/54 — 0.0% | 5/54 — 9.3% |
| >150 words | 26 | 0/26 — 0.0% | 0/26 — 0.0% | 0/26 — 0.0% |
| Priority human proxies | 23 | 1/23 — 4.3% | 0/23 — 0.0% | 3/23 — 13.0% |
| Application answers | 7 | 0/7 — 0.0% | 0/7 — 0.0% | 0/7 — 0.0% |
| Follow-ups | 4 | 0/4 — 0.0% | 0/4 — 0.0% | 0/4 — 0.0% |
| Learner essays | 18 | 0/18 — 0.0% | 0/18 — 0.0% | 0/18 — 0.0% |
| Learner reflections | 10 | 1/10 — 10.0% | 0/10 — 0.0% | 1/10 — 10.0% |
| Networking/referral asks | 2 | 0/2 — 0.0% | 0/2 — 0.0% | 1/2 — 50.0% |
| Practical forums | 16 | 0/16 — 0.0% | 0/16 — 0.0% | 0/16 — 0.0% |
| Scientific/encyclopedic | 6 | 0/6 — 0.0% | 0/6 — 0.0% | 0/6 — 0.0% |
| Narrative fiction | 6 | 1/6 — 16.7% | 0/6 — 0.0% | 1/6 — 16.7% |
| Thank-you notes | 10 | 1/10 — 10.0% | 0/10 — 0.0% | 2/10 — 20.0% |
| Other work letter | 1 | 0/1 — 0.0% | 0/1 — 0.0% | 0/1 — 0.0% |
| Source: PELIC | 40 | 1/40 — 2.5% | 0/40 — 0.0% | 2/40 — 5.0% |
| Source: Stevenson | 12 | 1/12 — 8.3% | 0/12 — 0.0% | 2/12 — 16.7% |
| Source: Wells | 6 | 1/6 — 16.7% | 0/6 — 0.0% | 1/6 — 16.7% |
| Source: Huxley | 6 | 0/6 — 0.0% | 0/6 — 0.0% | 0/6 — 0.0% |
| Source: Bicycles Stack Exchange | 8 | 0/8 — 0.0% | 0/8 — 0.0% | 0/8 — 0.0% |
| Source: Home Improvement Stack Exchange | 8 | 0/8 — 0.0% | 0/8 — 0.0% | 0/8 — 0.0% |

### Recorded first languages within PELIC

| Recorded L1 | N | Ours FP | ZeroGPT FP | Gate wrongful FAIL |
|---|---:|---:|---:|---:|
| Arabic | 8 | 0/8 | 0/8 | 0/8 |
| Chinese | 8 | 1/8 — 12.5% | 0/8 | 1/8 — 12.5% |
| Korean | 8 | 0/8 | 0/8 | 0/8 |
| Japanese | 3 | 0/3 | 0/3 | 0/3 |
| Spanish | 3 | 0/3 | 0/3 | 0/3 |
| Hebrew | 2 | 0/2 | 0/2 | 0/2 |
| Thai | 2 | 0/2 | 0/2 | 0/2 |
| French | 1 | 0/1 | 0/1 | 0/1 |
| Italian | 1 | 0/1 | 0/1 | 1/1 — 100% |
| Montenegrin | 1 | 0/1 | 0/1 | 0/1 |
| Portuguese | 1 | 0/1 | 0/1 | 0/1 |
| Russian | 1 | 0/1 | 0/1 | 0/1 |
| Suundi | 1 | 0/1 | 0/1 | 0/1 |

**Worst observed human domain/source at the 50% detector threshold: Wells narrative fiction, 1/6 (16.7%)**, versus ZeroGPT 0/6. The short-human FPR, 2/22 (9.1%), is more directly relevant operationally. At the gate, thank-you notes are 2/10 (20%); the still smaller referral group is 1/2 and Italian-L1 group 1/1. Those tiny denominators must be disclosed, not presented as reliable evidence of a language-specific population disparity. There is no basis to infer causation from L1.

Approximate 95% Wilson intervals: ours aggregate human FPR **3/80: 1.3–10.5%**; hard-AI sensitivity **35/40: 73.9–94.5%**; short-human FPR **2/22: 2.5–27.8%**. ZeroGPT's 0/80 human errors still has an upper Wilson bound of about **4.6%**. Source, prompt and derivative clustering weaken independent-sample interpretations further. Zero errors in one small stratum do not demonstrate absence of bias.

## Hard styles and generator provenance

| Hard-AI style | N | Ours caught | ZeroGPT caught | Gate caught |
|---|---:|---:|---:|---:|
| Natural, restrained | 25 | 21/25 — 84.0% | 23/25 — 92.0% | 22/25 — 88.0% |
| Humanize/paraphrase | 7 | 6/7 — 85.7% | 7/7 — 100% | 6/7 — 85.7% |
| Lowercase | 4 | 4/4 — 100% | 4/4 — 100% | 4/4 — 100% |
| Non-native imitation | 4 | 4/4 — 100% | 4/4 — 100% | 4/4 — 100% |

| Recorded generator, polishing excluded | N | Ours caught | ZeroGPT caught | Gate caught |
|---|---:|---:|---:|---:|
| `claude-sonnet-4-6` | 16 | 15/16 — 93.8% | 16/16 — 100% | 15/16 — 93.8% |
| `gemini-3.1-flash-lite` | 16 | 16/16 — 100% | 16/16 — 100% | 16/16 — 100% |
| `gpt-6.1-terra` | 16 | 14/16 — 87.5% | 16/16 — 100% | 16/16 — 100% |
| Reviewer `gpt-6.1-sol` | 17 | 11/17 — 64.7% | 15/17 — 88.2% | 11/17 — 64.7% |

Misses are concentrated in the reviewer's ordinary-message writing, but that is not proof of an intrinsic model-family weakness: model, prompt and writing-style effects are confounded, and four generator sources do not span today's frontier. Two restrained messages evade both detectors. Perfect results on four lowercase or four non-native-imitation examples do not establish universal robustness.

## Ten instructive local failures — primary round 3

Short quotes below are from the frozen texts. Scores are model outputs, not demonstrated calibrated probabilities of authorship. PASS on a generated text is a gate miss; FAIL on an authentic pre-2022 human text is a wrongful block.

| ID / provenance | Short quote | Words | Ours AI % | ZeroGPT AI % | Gate | Why it matters |
|---|---|---:|---:|---:|---|---|
| R3H013 — human, PELIC, Chinese L1 | “what was a hot day yestertaday before the heavy rain came up.” | 35 | 85.88 | 0.0 | FAIL | A genuine imperfect learner draft receives a confident false positive; short/ESL writing is not protected by the strong aggregate non-native result. |
| R3H047 — human, Stevenson thanks | “I do not know how to thank you for all your kind trouble in the matter of ‘The Sea-Cook’” | 57 | 53.83 | 0.0 | FAIL | Historical human gratitude is wrongly rejected in the priority short-message genre; its em-dash tells are soft, not the blocking condition. |
| R3H063 — human, Wells fiction | “Hitherto … I had felt a sustaining hope of ultimate escape” | 135 | 83.90 | 0.0 | FAIL | An authentic 1895 passage exposes the worst observed human source/domain. |
| R3H009 — human, PELIC thanks, Italian L1 | “I really appreciate your presentation this past Friday” | 89 | 36.60 | 0.0 | FAIL | A detector-correct human becomes a gate error; saved attribution singles out “Sincerely,”. Lowering the threshold is not a free gain. |
| R3S003 — raw reviewer AI, recruiter reply | “could you confirm whether the salary range is for the full five days?” | 51 | 11.93 | 100.0 | PASS | A concrete, ordinary recruiter response is missed even without a humanization instruction. |
| R3S005 — raw reviewer AI, networking ask | “No need to put me forward at this stage. I'm just trying to work out whether it makes sense to apply.” | 103 | 6.77 | 88.3 | PASS | The old round-2 networking example is fixed, but a fresh restrained example of the same practical use still passes. |
| R3S011 — restrained reviewer AI, referral permission | “I wouldn't call it a referral unless you're comfortable with that.” | 51 | 4.35 | 100.0 | PASS | Accurate social nuance and a narrow ask evade the local model, far below either operating threshold. |
| R3S017 — humanized reviewer AI, follow-up | “It was a separate attachment, so I'm not sure it ended up with the form.” | 92 | 14.07 | 100.0 | PASS | Specific operational follow-up survives a humanizing rewrite and the stricter gate. |
| R3S010 — restrained reviewer AI, recruiter correction | “I use Python for a couple of reports, but I'm not a software engineer.” | 50 | 0.50 | 0.0 | PASS | Both detectors miss this generated correction; the reference is not an oracle. |
| R3S008 — restrained reviewer AI, LinkedIn outreach | “You mentioned that new starters help check the ticket categories.” | 50 | 10.99 | 32.0 | PASS | Both detectors miss a plausible talk-follow-up inquiry; authentic modern human counterparts are still absent from this audit. |

Other saved errors, including mixed-provenance polishing and gate-only cases, are listed in `phase_b_tables.md` and retain full raw responses in `results.json`. No new samples were substituted after seeing these failures.

## Round 2 rerun — what changed on the same old texts

**Not unseen validation:** the builder's v7 training templates were partly informed by round 2's reported failures. This section measures regression behavior on old examples, not independent generalization. The saved ZeroGPT outputs are unchanged.

| Category | N | Ours at round 2 | Ours now | Saved ZeroGPT | Gate now |
|---|---:|---:|---:|---:|---:|
| Human native/proficient proxy — FP | 30 | 1/30 — 3.3% | 2/30 — 6.7% | 1/30 — 3.3% | 2/30 — 6.7% |
| Human non-native — FP | 30 | 0/30 | 0/30 | 0/30 | 0/30 |
| AI raw | 20 | 20/20 — 100% | 20/20 — 100% | 20/20 — 100% | 20/20 — 100% |
| AI casual/humanized | 30 | 26/30 — 86.7% | 29/30 — 96.7% | 29/30 — 96.7% | 29/30 — 96.7% |
| AI-polished human — separate | 10 | 5/10 — 50.0% | 4/10 — 40.0% | 6/10 — 60.0% | 5/10 — 50.0% |

| Matched-set metric | Ours at round 2 | Ours now | Saved ZeroGPT |
|---|---:|---:|---:|
| Fixed quality score | 9.40/10 | 9.73/10 | 9.80/10 |
| Accuracy excluding polishing | 105/110 — 95.5% | 107/110 — 97.3% | 108/110 — 98.2% |
| All-text accuracy, polishing positive | 110/120 — 91.7% | 111/120 — 92.5% | 114/120 — 95.0% |
| Priority human FPR | 0/13 | 0/13 | 0/13 |
| Generated professional-AI sensitivity | 35/37 — 94.6% | 37/37 — 100% | 36/37 — 97.3% |
| Hard professional-AI sensitivity | 19/21 — 90.5% | 21/21 — 100% | 20/21 — 95.2% |
| LinkedIn-style AI sensitivity | 6/7 — 85.7% | 7/7 — 100% | 7/7 — 100% |

The round-2 priority proxy uses the same earlier diagnostic genre set: cold outreach, follow-up/work email, reference requests, LinkedIn messages/posts, application answers and cover letters. It is broader than the round-3 seven-genre annotation; these slices are not directly interchangeable. Round 2 has **no 30–60-word samples**.

**Biggest observed improvement since round 2:** three additional hard-AI texts are caught at 50% (26/30 → 29/30), notably the networking message `R2S005`, **26.51%/PASS → 78.57%/FAIL**. `R2KH03`, a casual review, rises **8.74%/PASS → 55.04%/FAIL**. `R2S004`, an imperfect-English reference request, rises **34.49%/FAIL → 78.93%/FAIL**: a detector-threshold fix, not a new gate catch.

There is a human regression: Austen `R2H060` rises **32.67% → 63.26%**, becoming a false positive at 50%; it was already gate-blocked. Austen is now the worst old-set human source, **2/3 FP (66.7%)**, versus ZeroGPT 1/3. Broader old-set narrative fiction is 2/6 versus ZeroGPT 1/6. Other human domains have 0 observed detector FPs. These tiny literary strata do not establish population FPRs.

The gate improves from primary **104/110 to 107/110** correct, fixing the old networking and casual-review misses and releasing a genuine technical-forum question (`R2H052`, **39.20%/FAIL → 5.86%/PASS**). The remaining generated old-set narrative miss `R2KH06` still returns **2.08%/PASS**, versus ZeroGPT 100%. Hard-style catches are now casual 7/8, lowercase 7/7, non-native imitation 7/7, humanizing rewrites 8/8. Raw AI remains 20/20. Old-set parity is still **no** under the prespecified rule: accuracy and human specificity remain slightly below ZeroGPT despite tied hard-AI sensitivity.

## Limits, interpretation, and next improvement

1. **Round 3 is the decision set.** Round 2's development-informed gains demonstrate fixes on those examples, not a clean held-out win. No previous engine was evaluated on round 3, so differences between rounds cannot identify an engine-only effect.
2. **Genuine modern human job-search coverage is incomplete.** Historical letters and classroom tasks are labeled proxies, not authentic modern LinkedIn/recruiter communications. Era, genre and source mismatch can confound both error and success rates.
3. **Samples are clustered.** Forty non-native texts share PELIC; 24 native/proficient excerpts share three historical authors; prompt/profile families repeat. Polished samples share human parents and some humanized samples share raw-AI parents. Uncertainty is greater than a naive independent-text analysis suggests.
4. **Generator labels are not identity guarantees.** Different requested model IDs do not prove independence from builder training, and source novelty cannot prove absence from foundation-model pretraining. No claim of universal frontier-model coverage is warranted.
5. **Polishing is genuinely ambiguous.** A light edit is not interchangeable with wholly generated AI; its apparent sensitivity should not drive the primary score or be sold as human-authorship adjudication.
6. **Neither detector establishes authorship.** ZeroGPT also misses two restrained professional examples. Scores should not be used to accuse a writer, and PASS does not establish human origin. Automatic forced rewrites of genuine human messages can damage the owner's voice without improving truthfulness.

The most valuable next development experiment is matched, source-/author-/prompt-/generator-held-out short professional writing with an explicit human-FPR budget and an uncertainty policy. Include genuine recruiter corrections, referral-permission asks and attachment follow-ups, alongside real imperfect human thanks and short learner drafts. Do not train on these evaluation texts or tune to these ten quoted cases. Threshold cuts alone cannot separate the strongly inverted examples observed here.

## Reproducibility and files

- `texts.json`, `manifest.json`: unchanged frozen round-3 input and identity.
- `zerogpt.json`: unchanged phase-A reference answers.
- `results.json`: 160 round-3 rows with raw local answers and reused ZeroGPT results.
- `round2-results.json`: 120 current-service rows with round 2's reused reference answers; original round-2 `results.json` remains intact.
- `phase_b_summary.json`: primary and regression metrics, subgroup counts, versions, errors and matched old-set changes.
- `phase_b_tables.md`: complete generated tables, including both datasets' source/domain bias tables and polishing breakdowns.
- `phase_b_audit.json`: separately recomputed fixed-score components.
- `phase_b_evaluation.log`, `phase_b_analysis.log`: collection and audit logs.
- `evaluate_phase_b.py`, `analyze_phase_b.py`, `build_phase_b_tables.py`: evaluation/analysis scripts confined to this round folder.
- `phase_b_comparison.html`: local interactive comparison of the two datasets' fixed scores; no benchmark text is embedded.

**Evaluation complete. No detector edits, restarts, commits, or new external-detector queries were performed.**
