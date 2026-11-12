# Round 2: independent local AI-detector review

## Bottom line and message to the main agent

**Fresh-set quality score: ours 9.4/10; ZeroGPT 9.8/10.**

**Round-1-set rerun with the current service: ours 9.7/10 (unrounded 9.6667); ZeroGPT 10.0/10.**

**My overall judgment, separate from the fixed score: 8.5/10.** This is a useful, substantially improved local detector, but not yet demonstrated to match the reference or to be a reliable release gate for the owner's priority writing.

**Is ours at least as good as ZeroGPT on this set? no.**

**Is this a frontier-level AI-text detector? no.** The single biggest reason is remaining misses on naturally casual AI, including a fully generated LinkedIn networking message that passes the actual `/verify` gate. The fresh set's casual-style AI catch rate is 5/8, versus ZeroGPT's 8/8. This small sample does not establish a universal gap, but it does not support a frontier/parity claim either.

**Explicit user clarification for the main agent:** the primary use case is **casual-professional job-search communication: LinkedIn outreach, recruiter replies, networking and referral requests, follow-ups, and short application answers**. Optimizing generic essays, formal cover letters, or encyclopedic prose is not enough. Prioritize preserving normal human messages and detecting generated writing in those actual microgenres. This clarification arrived during the frozen evaluation; it did not change the texts, thresholds, or agreed score. The priority-use slices below are explicitly post-hoc diagnostics, not a replacement score.

**Single biggest improvement:** develop and validate a source/model-held-out, low-false-positive detector specifically for natural casual-professional writing, using matched genuine human and independently generated outreach/recruiter/referral/application messages. Preserve the improved non-native specificity. Do not solve the LinkedIn miss by globally lowering the threshold or reintroducing stock-phrase vetoes. Use other development data, not this audit set.

## Method and integrity

The fresh set was finalized and timestamped at **2026-11-12 04:43:00.506354 UTC**, before any detector call on it. `texts.json` includes the timestamp, labels, sources, text hashes, word counts and generation provenance. The final manifest's whole-file SHA-256 is:

`45f4711b900d4f2ddf4ee473aa5459edf10642be7aa4469b4e595d46e00ab247`

An initial generation-stage freeze was followed, before testing, by a source-only genre annotation audit and a final timestamp; both are visible in the scripts/logs. No sample was replaced or text edited using detector feedback. The final hash still matches. All 120 fresh text hashes are unique, and **none matches any round-1 text**. Round 1's original file also remains unchanged, with hash:

`fa465126e7b4e635f0cda33440424e24e93ecd74ae41cbc096aaa7229ca58221`

Fresh evaluation ran **04:43:00–04:56:12 UTC**; the complete round-1 rerun ran **04:56:12–05:09:28 UTC**. These are actual machine timestamps, not the date assumed in the original request.

For each text, the script called `/api/detect`, then `/verify`, then ZeroGPT. There was only one local request in flight. ZeroGPT requests were separated by at least **5.300 seconds after the previous request completed**, including across datasets and retries. There were two ZeroGPT retries and no local retries; all final endpoint checks succeeded. All three endpoints have **120/120 successful answers per dataset**, 720 successful final answers overall. No failed checks were silently treated as correct. No private owner writing was submitted: inputs were synthetic or public pre-2022 human prose, including publisher-anonymized learner submissions.

All 240 verification responses returned the same versions:

- **Engine:** `tropa-mini-ai-text-detector@10fecd08 + rp-modernbert-v6@1579161232-1794457937+stack@762af930`
- **Service:** `1`
- **Tells:** `3`

The before/after health outputs are saved in the result files. All 240 `/api/detect` outputs report `device: cuda`; initial health said CPU-loaded before the first check, which is not the device actually reported for detections. Every `/verify` response is complete and reports `from_store: false`. `/api/detect` and `/verify` scores agree on every text. This report tests the version returned by the service, not an assumed model or its implementation. I did not inspect or edit detector code/models, restart the service, or commit anything.

### Frozen scoring rules

- Ours: AI flag at `/api/detect` `ai_score >= 0.50`.
- ZeroGPT: AI flag at `fakePercentage >= 50`.
- Gate: report `/verify` FAIL separately; its default maximum is 30%, with its own tell policy. A block is not the same measurement as the 50% detector threshold.
- Fixed quality formula, unchanged from round 1 and identical for both detectors:
  `10 × (0.40 × human specificity + 0.20 × raw-AI sensitivity + 0.40 × casual/humanized-AI sensitivity)`.
- Polished human is a separate, mixed-provenance category and is excluded from that score. Overall accuracy is given both counting it as AI-derived positive and excluding it.
- Predeclared parity requires at least ZeroGPT's polishing-excluded accuracy, no higher human FPR, and no lower hard-AI sensitivity. Ours fails the accuracy and hard-AI conditions on fresh data; aggregate human FPR is tied.

`analyze.py` computes all tables; `audit.py` independently recomputes the scores directly from raw responses and checks hashes, completed statuses, pacing, changes from round 1 and the priority-use slices. Both agree.

## Fresh dataset

**120 texts, 74–392 words:** 60 humans, 50 wholly generated AI, and 10 AI-polished human texts. No source text from round 1 was reused.

Human sources, distinct from round 1's collections/authors/sites:

- **24 Cambridge Write & Improve ESL submissions**, released in BEA2019 v2.1, original uncorrected JSON. Includes job letters, recruitment-related writing, an academic reference request, informal messages and essays/stories. The publisher manually anonymized names and identifiers. The corpus establishes ESL provenance, but individual first languages are not released. I do not infer languages from names or locations. Its practical letters are prompt-based exercises, not verified real job applications.
- **6 literary excerpts from non-native-English authors:** two each from Kakuzo Okakura (Japanese L1), Rabindranath Tagore (Bengali L1), and Joseph Conrad (Polish L1), published 1906–1913. These provide several first-language backgrounds but are edited literary English, not representative learner writing. Editorial/translation assistance is a confound, especially for literary-philosophical texts.
- **8 LOCNESS native-English student essays**, released before 2022, uncorrected. Long originals were excerpted at complete opening paragraphs or sentence boundaries, not rewritten.
- **16 technical forum posts from 2018**, eight each from Open Source and Software Engineering Stack Exchange, with no post-2021 last edits in the acquired metadata. HTML was converted to plain text; examples with code blocks, URLs, contact markers and obvious sensitive content were excluded before any detector test. Individual first languages are unknown: this is an English-proficient/native-proxy stratum, not 16 verified native speakers.
- **6 native-English historical prose excerpts:** three each from Darwin and Austen. These test scientific/encyclopedic and literary prose but are not contemporary professional messages.

All are outside the brief's named exclusion collections. Public-domain hosts and the BEA redistribution infrastructure overlap broadly with round 1, but the actual corpora, books/authors and Stack Exchange sites are different. Novelty relative to these exclusions is **not** proof of absence from either detector's pretraining or other builder data.

AI provenance: **18 texts each from `claude-haiku-5-5`, `gemini-3-flash`, and `gpt-6-luna`, plus 6 written directly by reviewer `gpt-6.1-sol`**. These are recorded proxy/reviewer model identifiers, not independently authenticated underlying model identities. There were 12 successful batched proxy calls, no generation retries, no Fable calls and no Opus bulk generation. Raw prompts/responses are saved in `generation/`. The 30 hard cases preserve round 1's style counts: 8 casual, 8 humanize/paraphrase rewrites, 7 lowercase, 7 non-native imitations. Ten polishing operations have explicit human-parent links.

Limits in coverage matter: there are **no authentic human LinkedIn posts/messages**, no real recruiter inbox, and little genuinely modern native-human professional-email material. Source clusters, repeated task families, derivative text pairs, and small subgroup sizes preclude population-level claims. The requested 60-word minimum also excludes many real LinkedIn DMs and recruiter replies; these results do not validate sub-60-word use.

## Fresh-set results — primary

Counts are flagged-as-AI rates; lower is better for human rows, higher for AI rows. Gate counts are FAIL rates.

| Category | N | Ours ≥50% | ZeroGPT ≥50% | Ours `/verify` FAIL |
|---|---:|---:|---:|---:|
| Human native/proficient proxy | 30 | 1/30 — 3.3% | 1/30 — 3.3% | 3/30 — 10.0% |
| Human non-native | 30 | 0/30 — 0.0% | 0/30 — 0.0% | 0/30 — 0.0% |
| AI raw | 20 | 20/20 — 100.0% | 20/20 — 100.0% | 20/20 — 100.0% |
| AI casual or humanized | 30 | 26/30 — 86.7% | 29/30 — 96.7% | 27/30 — 90.0% |
| AI-polished human | 10 | 5/10 — 50.0% | 6/10 — 60.0% | 5/10 — 50.0% |

| Evaluation | Detector/gate | Correct / N | Accuracy | TP | TN | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| All, polishing counted positive | Ours | 110/120 | 91.7% | 51 | 59 | 1 | 9 |
| All, polishing counted positive | ZeroGPT | 114/120 | 95.0% | 55 | 59 | 1 | 5 |
| All, polishing counted positive | Ours `/verify` | 109/120 | 90.8% | 52 | 57 | 3 | 8 |
| Excluding polishing | Ours | 105/110 | 95.5% | 46 | 59 | 1 | 4 |
| Excluding polishing | ZeroGPT | 108/110 | 98.2% | 49 | 59 | 1 | 1 |
| Excluding polishing | Ours `/verify` | 104/110 | 94.5% | 47 | 57 | 3 | 3 |

The gate buys one additional hard-AI catch while wrongly blocking two additional humans. Those two extra human blocks are both **model scores between 30% and 50%**, not stock-closer rules. Unlike round 1's earlier service, the current gate does not block the tested conventional learner-letter closers.

### Hard styles and current models

| Hard-AI style | N | Ours caught | ZeroGPT caught | Gate caught |
|---|---:|---:|---:|---:|
| Naturally casual | 8 | 5/8 — 62.5% | 8/8 — 100.0% | 5/8 — 62.5% |
| Humanize/paraphrase | 8 | 8/8 — 100.0% | 8/8 — 100.0% | 8/8 — 100.0% |
| Lowercase | 7 | 7/7 — 100.0% | 7/7 — 100.0% | 7/7 — 100.0% |
| Non-native imitation | 7 | 6/7 — 85.7% | 6/7 — 85.7% | 7/7 — 100.0% |

| Recorded model; polishing excluded | N | Ours caught | ZeroGPT caught |
|---|---:|---:|---:|
| claude-haiku-5-5 | 14 | 12/14 — 85.7% | 14/14 — 100.0% |
| gemini-3-flash | 15 | 15/15 — 100.0% | 15/15 — 100.0% |
| gpt-6-luna | 15 | 15/15 — 100.0% | 15/15 — 100.0% |
| Reviewer gpt-6.1-sol | 6 | 4/6 — 66.7% | 5/6 — 83.3% |

The remaining fresh misses are **not** the lowercase group. They are three natural-casual examples plus one non-native-style reference request. The gate catches the latter and still misses all three casual examples. Do not describe the 100% paraphrase result as proof of robustness to all humanization techniques; only eight particular transformations were tested.

### Priority use: casual-professional job search

These slices were defined after the user's clarification and are diagnostic only. The professional AI slice comprises cold outreach, follow-up/work email, reference requests, LinkedIn messages/posts, application answers and cover letters; polishing is excluded. The human subset contains only the matching stored genres actually available, all Write & Improve proxies here.

| Fresh priority slice | N | Ours flags | ZeroGPT flags | Gate blocks |
|---|---:|---:|---:|---:|
| Human job/work/application proxies — false positives | 13 | 0/13 — 0.0% | 0/13 — 0.0% | 0/13 — 0.0% |
| All professional AI — sensitivity | 37 | 35/37 — 94.6% | 36/37 — 97.3% | 36/37 — 97.3% |
| Hard-style professional AI — sensitivity | 21 | 19/21 — 90.5% | 20/21 — 95.2% | 20/21 — 95.2% |
| LinkedIn-style posts/messages — sensitivity | 7 | 6/7 — 85.7% | 7/7 — 100.0% | 6/7 — 85.7% |

**The decisive directly relevant example is R2S005:** a natural, plausible networking message asking an old colleague about a vacancy. Ours scores it **26.51% and returns PASS**; ZeroGPT scores it **82%**. R2S004, an imperfect-English reference request, scores 34.49% locally and 8.7% at ZeroGPT: both miss at 50%, while the local gate catches it. Thus the gate ties ZeroGPT's aggregate professional sensitivity here through *different* decisions; that does not make their errors interchangeable for the owner.

Only **one** of the fresh professional hard examples is the explicitly `casual` subtype; the rest are predominantly lowercase, non-native imitation or paraphrase. That one is R2S005 and is missed. It would be irresponsible to report a meaningful 0/1 population estimate; instead, it exposes a coverage gap in this evaluation and a high-priority failure to test more deeply. The broader casual group includes reviews and fiction, so its 5/8 result is not itself a job-search sensitivity estimate.

Round 1's existing professional AI examples now score 40/40 caught, including 23/23 hard and 7/7 LinkedIn-style, for both detectors and the gate. That improvement is real on those saved examples, but **the new LinkedIn miss is why the fresh set matters**. The old set may have informed development and is not an unseen validation set.

## Human bias table — fresh set

False positives at the stated detector thresholds; the last column is the gate's wrongful-block rate. Strata overlap and must not be summed. Short = at most 150 words; long = more than 150 words, fixed in the analysis plan. Formal/casual are source/genre annotations, not inferred English proficiency.

| Human subgroup | N | Ours FP | ZeroGPT FP | Gate wrongful block |
|---|---:|---:|---:|---:|
| Native/proficient proxy | 30 | 1/30 — 3.3% | 1/30 — 3.3% | 3/30 — 10.0% |
| Non-native | 30 | 0/30 — 0.0% | 0/30 — 0.0% | 0/30 — 0.0% |
| Formal | 38 | 1/38 — 2.6% | 1/38 — 2.6% | 2/38 — 5.3% |
| Casual | 22 | 0/22 — 0.0% | 0/22 — 0.0% | 1/22 — 4.5% |
| Short, ≤150 words | 27 | 0/27 — 0.0% | 1/27 — 3.7% | 2/27 — 7.4% |
| Long, >150 words | 33 | 1/33 — 3.0% | 0/33 — 0.0% | 1/33 — 3.0% |
| Academic essays | 8 | 0/8 — 0.0% | 0/8 — 0.0% | 0/8 — 0.0% |
| Application answer | 1 | 0/1 — 0.0% | 0/1 — 0.0% | 0/1 — 0.0% |
| Casual messages | 4 | 0/4 — 0.0% | 0/4 — 0.0% | 0/4 — 0.0% |
| Cover letters | 9 | 0/9 — 0.0% | 0/9 — 0.0% | 0/9 — 0.0% |
| Cultural essays | 2 | 0/2 — 0.0% | 0/2 — 0.0% | 0/2 — 0.0% |
| Other essays | 3 | 0/3 — 0.0% | 0/3 — 0.0% | 0/3 — 0.0% |
| Inquiry letter | 1 | 0/1 — 0.0% | 0/1 — 0.0% | 0/1 — 0.0% |
| Philosophical essays | 2 | 0/2 — 0.0% | 0/2 — 0.0% | 0/2 — 0.0% |
| Other professional letter | 1 | 0/1 — 0.0% | 0/1 — 0.0% | 0/1 — 0.0% |
| Reference request | 1 | 0/1 — 0.0% | 0/1 — 0.0% | 0/1 — 0.0% |
| Review | 1 | 0/1 — 0.0% | 0/1 — 0.0% | 0/1 — 0.0% |
| Scientific/encyclopedic | 3 | 0/3 — 0.0% | 0/3 — 0.0% | 0/3 — 0.0% |
| Narrative fiction, stored `short_story` | 6 | 1/6 — 16.7% | 1/6 — 16.7% | 2/6 — 33.3% |
| Technical forums | 16 | 0/16 — 0.0% | 0/16 — 0.0% | 1/16 — 6.3% |
| Work emails/letters | 2 | 0/2 — 0.0% | 0/2 — 0.0% | 0/2 — 0.0% |
| **Source: Jane Austen excerpts** | **3** | **1/3 — 33.3%** | **1/3 — 33.3%** | **2/3 — 66.7%** |
| Source: Write & Improve ESL | 24 | 0/24 — 0.0% | 0/24 — 0.0% | 0/24 — 0.0% |
| Source: LOCNESS | 8 | 0/8 — 0.0% | 0/8 — 0.0% | 0/8 — 0.0% |
| Source: Okakura / Tagore / Conrad, combined | 6 | 0/6 — 0.0% | 0/6 — 0.0% | 0/6 — 0.0% |
| Source: Darwin | 3 | 0/3 — 0.0% | 0/3 — 0.0% | 0/3 — 0.0% |

**Worst observed human subgroup for ours is Austen-source fiction: 1/3 false positives (33.3%)**, versus ZeroGPT's 1/3 on a *different* excerpt. At the broader domain level it is narrative fiction, 1/6 (16.7%). The tiny source sample does not establish an elevated population FPR. The two local gate-only human errors are an Austen excerpt and a genuine 2018 technical-forum question.

The non-native result is encouraging but is not proof of bias-free behavior. Approximate 95% Wilson intervals: human FPR 1/60 → **0.3–8.9%**; non-native FPR 0/30 → **0–11.4%**; hard-AI catch 26/30 → **70.3–94.7%**; natural-casual catch 5/8 → **30.6–86.3%**. Source/task clustering further weakens independent-sample interpretations. First-language-specific learner error rates cannot be estimated from the available metadata.

## Ten instructive local failure cases

Scores are model outputs, not demonstrated calibrated authorship probabilities. Quotations are short excerpts from the frozen set. Polishing cases are misses only under the declared AI-derived-positive convention.

| Case | Short quote | Ours AI % | ZeroGPT AI % | Gate | Why it matters |
|---|---|---:|---:|---|---|
| **R2S005**, generated LinkedIn networking request | “do you know anyone on the service team … there's a repair coordinator opening” | 26.51 | 82.0 | PASS | Directly matches the owner's casual-professional use. Concrete, restrained networking prose slips through both local thresholds. |
| **R2S004**, generated non-native-style reference request | “Would you be comfortable to be my reference?” | 34.49 | 8.7 | FAIL | Both detectors miss at 50%; ours' stricter gate is useful here. Do not describe this as a gate miss or a ZeroGPT success. |
| **R2KH03**, Haiku casual review | “Bought this mechanical kitchen timer … for about four quid” | 8.74 | 100.0 | PASS | Natural everyday detail and uneven rhythm remain a weak point, beyond simple lowercase styling. |
| **R2KH06**, Haiku casual fiction | “It smelled of his car, cold coffee and the peppermints he keeps” | 7.04 | 100.0 | PASS | A low-scoring generated narrative miss; a small universal threshold reduction would not fix it. |
| **R2H059**, human Austen, 1813 | “Elizabeth found herself quite equal to the scene” | 73.76 | 0.0 | FAIL | The single fresh detector false positive. Historical literary prose is not proof of machine authorship. |
| **R2H052**, human technical forum, 2018 | “why I have to install a server on top of a server?” | 39.20 | 0.0 | FAIL | Detector is correct at 50%, but the gate blocks an ordinary real-human short question. Threshold trade-off, not a stock-phrase veto. |
| **R2H060**, human Austen, 1813 | “I never could perceive anything extraordinary in them” | 32.67 | 74.3 | FAIL | Local detector correctly clears it, local gate does not; ZeroGPT also makes a human false positive. Same source, different errors. |
| **R2GP02**, Gemini-polished human art essay | “Contemporary art is a very controversial issue today” | 20.44 | 94.3 | PASS | Local note hints at a humanizer, but the final gate passes. Polishing policy needs to be explicit, not inferred from a single AI %. |
| **R2GP03**, Gemini-polished human Austen excerpt | “Upon his return, Mr. Collins highly gratified Mrs. Bennet” | 17.86 | 100.0 | PASS | Stylistic editing is not consistently identified; a soft construction tell is present but correctly not treated as conclusive alone. |
| **H047**, round-1 historical human cover letter, now a regression | “In a word, I have outgrown my present position” | 61.68 | 0.0 | FAIL | Relevant job-search genre. Improved average specificity still introduces a new confidently wrong human decision on old data. |

Other polishing misses are R2KP03 (1.46%; ZeroGPT 47.9%), R2LP02 (16.27%; ZeroGPT 0%), and R2KP04 (16.46%; ZeroGPT 0%). Neither detector consistently identifies all light editing. Detecting polishing is not equivalent to proving a whole text was generated, which is why those outcomes do not affect the primary score.

In the old-set rerun, S006 is now missed at 2.94% and passes; KH04 is missed at 43.92% but the gate catches it. Both were previously detector successes. The service's raw class vectors are saved, but a combiner's final score should not be judged solely by one member's probabilities. I make no claim about the internal scoring formula from those vectors.

## Round-1-set rerun — comparison, not fresh validation

These are new responses from the current service and newly submitted ZeroGPT requests, not copied prior scores.

| Category | N | Current ours ≥50% | Current ZeroGPT ≥50% | Current gate FAIL |
|---|---:|---:|---:|---:|
| Human native/proficient proxy | 30 | 1/30 — 3.3% | 0/30 — 0.0% | 1/30 — 3.3% |
| Human non-native | 30 | 0/30 — 0.0% | 0/30 — 0.0% | 0/30 — 0.0% |
| AI raw | 20 | 20/20 — 100.0% | 20/20 — 100.0% | 20/20 — 100.0% |
| AI casual/humanized | 30 | 28/30 — 93.3% | 30/30 — 100.0% | 29/30 — 96.7% |
| AI-polished human | 10 | 8/10 — 80.0% | 9/10 — 90.0% | 8/10 — 80.0% |

| Evaluation | Detector/gate | Correct / N | Accuracy | TP | TN | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| All, polishing counted positive | Ours | 115/120 | 95.8% | 56 | 59 | 1 | 4 |
| All, polishing counted positive | ZeroGPT | 119/120 | 99.2% | 59 | 60 | 0 | 1 |
| All, polishing counted positive | Gate | 116/120 | 96.7% | 57 | 59 | 1 | 3 |
| Excluding polishing | Ours | 107/110 | 97.3% | 48 | 59 | 1 | 2 |
| Excluding polishing | ZeroGPT | 110/110 | 100.0% | 50 | 60 | 0 | 0 |
| Excluding polishing | Gate | 108/110 | 98.2% | 49 | 59 | 1 | 1 |

On the **same old texts**, ours improves from **108/120 to 115/120** overall and from **100/110 to 107/110** excluding polishing. Human false positives fall **5 → 1**, raw sensitivity rises **19/20 → 20/20**, and hard sensitivity rises **26/30 → 28/30**. The gate improves from **106/120 to 116/120**. ZeroGPT's binary decisions are unchanged. These paired comparisons, not differences between dissimilar fresh sets, establish the observed improvement.

Ours fixes ten old errors and introduces three new ones: **H047** (human cover-letter FP), **S006** (AI fiction FN), **KH04** (AI forum FN at 50%; gate catches). Gate changes fix twelve old errors and introduce two new ones, H047 and S006. All three original non-native complaint false positives and the prior stock-closer gate-only false positives are gone on the tested examples. That is a substantial advance, not a reason to ignore the new regressions.

Old-set human bias overview:

| Old-set human subgroup | N | Current ours FP | Current ZeroGPT FP | Current gate block |
|---|---:|---:|---:|---:|
| Native/proficient proxy | 30 | 1/30 — 3.3% | 0/30 — 0.0% | 1/30 — 3.3% |
| Non-native FCE | 30 | 0/30 — 0.0% | 0/30 — 0.0% | 0/30 — 0.0% |
| Short ≤150 words | 14 | 0/14 — 0.0% | 0/14 — 0.0% | 0/14 — 0.0% |
| Long >150 words | 46 | 1/46 — 2.2% | 0/46 — 0.0% | 1/46 — 2.2% |
| Cover-letter domain | 5 | 1/5 — 20.0% | 0/5 — 0.0% | 1/5 — 20.0% |
| Other human domains combined | 55 | 0/55 — 0.0% | 0/55 — 0.0% | 0/55 — 0.0% |

## Scores, judgment and uncertainty

Fresh fixed formula:

- Ours: `10 × (0.40 × 59/60 + 0.20 × 20/20 + 0.40 × 26/30) = 9.4000` → **9.4/10**.
- ZeroGPT: `10 × (0.40 × 59/60 + 0.20 × 20/20 + 0.40 × 29/30) = 9.8000` → **9.8/10**.

Round-1 rerun:

- Ours: `10 × (0.40 × 59/60 + 0.20 × 20/20 + 0.40 × 28/30) = 9.6667` → **9.7/10**.
- ZeroGPT: `10 × (0.40 × 60/60 + 0.20 × 20/20 + 0.40 × 30/30) = 10.0000` → **10.0/10**.
- Original round-1 ours was 9.0333; the old-set diagnostic improvement is +0.6333 points. Fresh 9.4 versus original-round-1 9.0 is **not** a paired improvement estimate.

The fixed score rewards excellent fresh human preservation (59/60) and raw AI sensitivity (20/20), with solid but inferior hard-AI sensitivity (26/30). ZeroGPT ties human specificity and raw sensitivity, and catches three additional hard examples. Its 9.8 is a reference on this sample, not universal near-perfection: it also falsely flags Austen, misses a non-native-style AI reference request and misses four polishing cases. Ours has one non-polished decision uniquely correct versus four uniquely correct for ZeroGPT; exact paired McNemar p = **0.375**. On the old set these are zero versus three, p = **0.25**. Small samples and clustering mean neither establishes a statistically secure population ranking or equivalence. The literal set-specific parity answer remains no under the declared rule.

**My separate overall judgment is 8.5/10**, deliberately below the formula's 9.4. The current engine is materially better than the original and often useful as private feedback, especially with the non-native and lowercase improvements. But its most natural casual examples are substantially harder than its paraphrases; it passes a fresh generated LinkedIn request; its gate still wrongfully blocks some genuine short writing; and this evaluation lacks authentic human LinkedIn/recruiter messages or sub-60-word examples. Those limitations prevent treating the high aggregate score as a validated fit for the owner's principal use. This judgment is not substituted into either comparable numerical score.

No additional public detectors were used. I did not attempt login-required tools or bypass browser verification. Consequently, “not frontier-level” is based on failure of the brief's minimum conditions and the observed reference gap, not a claimed exhaustive benchmark of every commercial detector.

## Recommended next step

Build a **separate development and held-out audit track for casual-professional job search**: authentic consented/anonymized human networking requests, recruiter replies, referral/reference requests and form answers, matched to independently generated samples from several current models. Include fluent and non-native humans, lowercase and ordinary mixed case, terse and mildly rambling messages, and lengths **below 60 words** as well as 60–250. Split by human source, prompt family, derivative parent and generator model. Select/calibrate for low human wrongful-block rates, and assess AI recall at that constraint. Keep polishing as an explicit separate policy/output rather than silently redefining it as generated authorship.

For now, use the service as an advisory check, not proof of authorship or a guarantee that a PASS sounds human to every reader. Do not auto-rewrite facts or a person's voice merely because a counterfactual highlight changes a score. The final improvement target is reliable behavior on the owner's actual messages, not merely a higher essay benchmark.

## Files and reproduction

Everything created for this review is in this round-2 folder:

- `texts.json`, `manifest.json`: final frozen fresh set and integrity record.
- `results.json`: all fresh raw answers, retries and health snapshots.
- `round1-results.json`: complete new answers for the unchanged old set.
- `summary.json`, `audit.json`: computed tables, raw-error details and independent integrity/priority checks.
- `ANALYSIS_PLAN.md`: rules written before detection.
- `acquire.py`, `build_humans.py`, `generate_ai.py`, `freeze.py`, `evaluate.py`, `analyze.py`, `audit.py`: source/generation/evaluation/analysis scripts.
- `downloads/`: separately contained public source files, never executed; `generation/`: saved synthetic generation requests/responses; `self_ai.json`: reviewer-written samples.
- `acquisition.log`, `generation.log`, `evaluation.log`, `analysis.log`, `audit.log`: progress and audit outputs.
- `comparison.html` and its builder: final compact visual comparison; `interim.html`: explicitly provisional snapshot previously shown to the user.

Recompute analysis only with `python3 -I analyze.py` and `python3 -I audit.py` using their absolute paths. Do not rerun generation/freeze over the finished set. Detection reruns contact the live services, consume resources and may test a different version; the saved outputs are the evidence for this report.
