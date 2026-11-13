# The process

How this detector was built, in order — including what didn't work. Numbers are from the held-out tests, never from
training data. Updated each round.

## Round 0 — pick an engine (2026-11-11)

Six open detectors were scored on 743 texts (554 human, 66 fresh AI from 8 current models, 32 LLM-"humanized",
91 GPT-4-polished human essays):

| Engine | AUROC (human vs raw AI) |
|---|---|
| **tropa-mini** (DeBERTa-v3-large, 4 classes) | **0.990** |
| desklib v1.01 | 0.890 |
| fakespot RoBERTa | 0.764 |
| Binoculars on a Qwen2.5-1.5B pair | 0.597 |
| Fast-DetectGPT (same pair) | 0.557 |
| TMR RoBERTa | 0.479 |

A stacked combination did *worse* than tropa-mini alone, so it ran alone.

**Finding 1 — read the classes, not the card.** tropa-mini's suggested score `1 − P(human)` flagged 49 % of real
college essays (its "humanized" and "AI-edited" classes fire on polished human writing). Plain `P(ai)` flagged 3 %,
but let half of LLM-humanized text through. Compromise: `P(ai)`, or `1 − P(human)` when the model is ≥ 99 % sure
the text isn't human.

**Finding 2 — short texts need a different highlight.** For a 120-word email a 150-word window *is* the whole text,
so every sentence got the same score. Re-scoring the text once without each sentence shows which sentences drive
the verdict.

**Finding 3 — fp16 is slower on this GPU.** On a GTX 1660 Ti (no tensor cores) fp16 matmuls ran at 0.6 TFLOPS vs
4.4 in fp32. Switching the service to fp32 made checks ~3.6× faster.

## Round 1 — the first independent review

GPT-6.1 Sol built 120 fresh texts (pre-2022 human letters and learner essays from new sources; AI from several
current models, including casual, lowercase, non-native-imitating and humanized styles) and ran both detectors:
**ours 90.0 % vs ZeroGPT 99.2 %, score 9.0/10, "not at least as good as ZeroGPT".** Its three fixes: calibrate the
class mapping, cover learner-style and understated professional AI, and stop ordinary closers ("I look forward to
hearing from you") from failing human emails. The last one was fixed the same hour.

Also learned: ZeroGPT in 2026 is much better than in 2023 studies (Liang et al. measured 53 % false positives on
TOEFL essays; today it was 0 %), but it only caught 60 % of real StealthGPT-humanized text.

## Round 2 — the first fine-tune and the bias it taught us (k3)

Full fine-tune of tropa-mini on 18k texts (learner essays, Jabarian humanized pairs, emails, reddit, stories, 1,350
newly generated texts in email / application-answer genres) on a free Kaggle T4, ~1 h. Validation looked perfect
(AUROC 0.9998) — and the held-out bias report showed why validation isn't enough:

- humanized AI caught: 48 % → **97 %** (ZeroGPT 77 %), modern Claude/GPT/Gemini: ~75 % → **98 %**;
- but human **academic** writing flagged: 3 % → **48 %**, **encyclopedic** 19 % → 39 %.

The model had learned *formal = AI*, because every formal text in its training data was AI. **Not deployed.**

## Round 3 — domain-balanced data, and a second architecture

- +13,855 texts from MAGE, COLING-2025 MGT and HC3 where **every domain has both human and AI text** (76
  domain×label groups), plus a 1,544-text dev split per domain.
- Checkpoints chosen by **0.5·(1 − worst-domain false-positive rate) + 0.5·catch rate**, so one bad domain can't
  hide behind a good average.
- A second architecture, ModernBERT-large (395M), trained on Runpod (A40 / A6000, $0.59/h cap, scored on the pod).
- Runpod lessons: community GPUs were sold out all evening; requiring a CUDA ≥ 12.8 host driver hid every card;
  the CUDA 12.4 image needed a clean, pinned venv. Each failed attempt cost ~$0.02 thanks to the guards.

Result: better, still not good enough alone. v4 (DeBERTa) caught 98 % of AI text but still flagged 29 % of academic
writing; ModernBERT (mb1, mb5) flagged less but missed more. A plain average of two models didn't fix it either.

## Round 4 — a learned combiner, cross-fitted

Each model outputs four class probabilities (human / AI / AI-edited / humanized). A small logistic regression reads
three log-odds features per model and decides; human texts are weighted per domain so no domain dominates the fit.
To keep the numbers honest it is **cross-fitted**: split the 4,257 held-out texts in two by hash, fit on one half,
score the other, then swap. Every reported number is out-of-fold.

- tropa-mini + mb5: humans flagged 1.6 %, worst group 2.9 %, humanized AI 87.7 % (tropa-mini alone: 43 %) —
  278 vs ZeroGPT's 280 on the shared texts. Deployed (2 models, 1.3 s per 250 words on the 1660 Ti).
- The misses had one theme: **encyclopedic and clinical AI prose** ("Photosynthesis is the biological process…")
  from a 2026 benchmark, only 75–88 % caught. The training data had far more human Wikipedia/PubMed than AI text
  in that style, so the models had learned *encyclopedic = human*.
- A DeBERTa v5 flagged 21 % of native human text alone and added nothing to the combiner: dropped.

## Round 5 — matched pairs (v6)

`train/build_expo.py`: 900 real Wikipedia openings and 900 real PubMed abstracts (all pre-2019), and for **the same
topics** an AI-written opening/abstract from nine current models (markdown stripped so formatting is never the
tell). 3,576 texts, balanced per domain, deduplicated against every test set.

ModernBERT v6 on that data (43 min on an A40, $0.43):

| | alone | combined with tropa-mini (cross-fitted) |
|---|---|---|
| vs ZeroGPT on its 291 texts | 278 vs 280 | **281 vs 280** |
| human texts flagged | 3.6 % | **1.2 %** |
| worst human group | 6.9 % | **2.4 %** |
| humanized AI caught | 89 % | 88 % |
| encyclopedic/clinical AI caught | | **98 %** (was 88 %) |

All four parts of the bar met; deployed 2026-11-11 23:40. 281 vs 280 is parity rather than a clear win (one text),
which is exactly why the independent review builds its own fresh set.

Also running: open-weights AI text (Qwen, gpt-oss-20b, Nemotron, MiniMax, Gemma, …) from free API tiers, because the
combiner catches 98–100 % of Claude/GPT/Gemini text but far less from open-model families. Every model that
generates training text is listed in its row's source; none of the generated texts are published.

## Round 2 of the independent review

GPT-6.1 Sol built a fresh 120-text set (Cambridge Write & Improve learner writing, LOCNESS, 2018 Stack Exchange,
public-domain prose from non-native and native authors; AI from Claude Haiku, Gemini, GPT-6 and itself, many in hard
styles) and re-ran round 1's set. [Full report](../eval/sol/round-2/REPORT.md).

| | ours | ZeroGPT |
|---|---|---|
| fresh set, fixed score | **9.4** | 9.8 |
| round-1 set again | 9.7 (was 9.0) | 10.0 |
| human texts flagged (fresh) | 1/60 | 1/60 |
| non-native writers flagged | 0/30 | 0/30 |
| raw AI caught | 20/20 | 20/20 |
| casual / humanized AI caught | 26/30 | 29/30 |

The bias problem is gone on this set; the gap is **naturally casual AI writing**: 5 of 8 caught vs ZeroGPT's 8 of 8,
including a generated LinkedIn networking message that scored 27 % and passed the gate. That is exactly the writing
this detector exists for (outreach, recruiter replies, follow-ups, short application answers), so it is the next target.

## Round 6 — casual job-search messages (v7)

The writing this detector exists for — LinkedIn outreach, recruiter replies, follow-ups, short application answers —
had almost no matched human/AI data. The owner's own email couldn't be the human side (much of it is AI-assisted), so:

- **Human, 4,120 texts, 2008–2020:** Hacker News "Who wants to be hired?" intros, Stack Exchange workplace /
  freelancing / PM posts, 2017 brand support replies on Twitter, FOIA work email, Ubuntu IRC, and Reddit career
  subreddits (2015–2020, via the Arctic Shift archive). Kaggle's "cover letter" / "recruiter" / "Slack" sets were all
  2023+ and mostly LLM-made: rejected. (`train/build_casual_human.py`, `train/build_reddit_career.py`)
- **AI, 3,661 texts (548 humanized):** for 2,400 of the human texts a model writes its own text of the same kind,
  same situation, same length (it sees only the first ~15 words and is told not to continue them), plus 800 of the
  owner's genres in mostly natural, restrained styles. (`train/build_casual.py`)
- Plus 3,145 open-weights texts (Groq, OpenRouter, NVIDIA free endpoints).

v7 alone flags too much human text on the test sets (8.8 % overall, 16 % of blog/other, 11 % of native-English writing). But on the 750 casual texts held back from its
training it catches **97.5 %** of AI at **1 %** false positives — and the combiner, fitted only on the test sets,
was throwing that away (54 %). Refitted with the held-back casual texts as their own domain (still cross-fitted,
`eval/stack_casual.py`), the best pair drops tropa-mini altogether:

| combiner | test sets: humans flagged / worst group / humanized / vs ZeroGPT | held-back casual: AI caught / your genres / humans flagged |
|---|---|---|
| v0 + v6 (was live) | 1.3 % / 2.9 % / 90 % / 281 vs 280 | 62 % / 86 % / 1.5 % |
| **v6 + v7 (live)** | **1.9 % / 3.8 % / 88 % / 281 vs 280** | **86 % / 97 % / 0 %** |

Caveat: the held-back casual texts come from the same sources and generators as v7's training data, so those numbers
are optimistic; the independent round-3 set (different models, different human sources, a third of it the owner's
genres) is the real test. The verify tests' "short human email" fixture turned out to be AI-written (it now reads
35 %); the test now uses 10 real pre-ChatGPT human texts instead (≥ 9 must pass).

## Round 3 of the independent review

Two-phase this time: GPT-6.1 Sol built and froze 160 fresh texts and collected ZeroGPT's answers while v7 trained,
then ran our service once v6 + v7 was deployed. 92 of the 160 are the owner's genres; 48 are 30–60 words. Its AI
texts come from models none of our training data used (GPT-6.1 Terra, Gemini 3.1 Flash-Lite, Claude Sonnet 4.6 and
Sol itself). [Full report](../eval/sol/round-3/REPORT.md).

| | ours | ZeroGPT |
|---|---|---|
| fixed score, fresh round-3 set | **9.0** | 9.8 |
| round-2 set again (no longer unseen) | 9.7 (was 9.4) | 9.8 |
| human texts flagged | 3/80 | 0/80 |
| raw AI caught | 21/25 | 25/25 |
| casual / humanized AI caught | 35/40 | 38/40 |
| owner's genres, AI caught | 50/57 | 55/57 |
| 30–60-word humans flagged | 2/22 | 0/22 |

What it says: casual-message detection improved (round 2's missed networking message is now caught at 79 %), but
the fresh set exposes two costs of the v6 + v7 engine — raw AI from unseen models (GPT-6.1 Sol's own writing: 11 of 17
caught) and a few short human messages (thank-you notes) flagged. Not yet ≥ ZeroGPT; not frontier.

## Round 7 — a length-aware combiner, and a regression benchmark

The combiner now also sees the text's length (log words, and length × each member feature), so it can weigh the two
models differently for a 40-word message than for a 400-word essay. Cross-fitted on the test sets + held-back casual
texts: short messages (≤ 80 words) 73 % → **85 %** of AI caught, short humans flagged 0.8 % → 1.6 %, every bar item
still met. No new training.

`eval/bench_sol.py` re-scores all 400 texts of the three finished review rounds (no longer unseen, so this is a
regression benchmark) and reproduces the reviewer's numbers exactly:

| engine | round 1 | round 2 | round 3 | all 400 | short AI caught (r3) | short humans flagged (r3) |
|---|---|---|---|---|---|---|
| v0 + v6 (round 2's) | 9.7 | 9.4 | **8.2** | 9.0 | 6/21 | 2/22 |
| v6 + v7 (round 3's) | 9.7 | 9.7 | 9.0 | 9.4 | 16/21 | 2/22 |
| **v6 + v7 length-aware (live)** | 9.6 | 9.7 | 9.1 | 9.4 | **17/21** | **1/22** |
| ZeroGPT | 10.0 | 9.8 | 9.8 | 9.9 | 19/21 | 0/22 |

So the round 2 → 3 drop (9.4 → 9.0) was the harder set, not a worse model: on round 3's texts the round-2 engine
scores 8.2. The length-aware combiner trades two long human texts flagged (8/200 vs 6/200) for better short-message
detection — the owner's use case. Scoring all 400 texts with three models takes 92 s on the 1660 Ti; a Runpod pod
would be ~15–20 min, nearly all of it uploading the models over a 36 Mbps uplink.


## Round 8 — more model families (v8)

Round 3's misses were text from models we had never trained on, so v8 adds breadth without new generation cost:
public datasets (2026 DeepSeek V4, Grok 4.6, GLM 5.3, Qwen 3.8 Max, Gemini 3.7, GPT-5.6 Luna Pro and Nemotron
texts; real GPT-3.5/GPT-4 chat replies from WildChat; Llama-2, Mixtral, Mistral, Gemma, zephyr and tulu answers paired
with human ones from Beemo), a casual pass from 12 more proxy models, free-tier open models, and ~2,900 of the owner's
own Claude.ai replies (assistant text only, scrubbed: `train/build_transcripts.py`). A packaging-time sanitizer
(`train/sanitize.py`, 21 test cases, `train/audit_data.py`) drops contact details and leftover code / tables /
diagrams with the same rules for both labels; its review caught three bugs (prices read as LaTeX, `|`-separated human
posts read as tables, bracketed asides read as placeholders).

v8 alone: 281 vs ZeroGPT 280 on the shared test texts — the first single model of ours to pass it. With tropa-mini in
a length-aware combiner:

| engine | test sets: vs ZeroGPT / humans flagged / worst group | held-back casual AI caught | review texts, all 400 | round 3's set |
|---|---|---|---|---|
| v6 + v7 length-aware (was live) | 281 vs 280 / 1.8 % / 4.1 % | 92 % | 9.4 | 9.1 |
| **v0 + v8 length-aware (live)** | **285 vs 280 / 1.3 % / 2.2 %** | **95 %** | **9.8** | **9.8** |
| ZeroGPT | 280 | — | 9.9 | 9.8 |

On the 400 review texts it now catches every raw-AI text (65/65), more casual/humanized AI than ZeroGPT (98 vs 97 of
100) and more of the owner's genres (56 vs 55 of 57); the remaining gap is human texts flagged (4 vs 1 of 200).
Caveat: those rounds are no longer unseen (round 3's generator choice informed v8's model list), so a fresh review
round is the real test. Cost: 72 min on an A40, $0.71 — plus $0.04 for a second pod a hand-written retry loop started
by mistake after the first finished (it looked for "pod up" in the last 60 log lines, which scoring had pushed out);
caught and deleted within 4 minutes.

## Round 9 — public benchmarks find the next gaps

Before spending another review round, the live engine was run on three published benchmarks it never trained on
([public benchmarks](public-benchmarks.md)). OUTFOX: 99 % (every detector-evasion essay caught). Ghostbuster: 92 %.
DetectRL: 82 % vs ZeroGPT 96 % on the shared sample — the misses are 2023 Claude-instant and PaLM text and the
back-translation attack, while AUROC stays ≥ 0.99. That is exactly what a benchmark should do: find where it breaks.
Fix in progress (v9): the benchmarks' training splits for those families + locally back-translated text.

## Round 10 — v9: closing the public-benchmark gaps (trained, not deployed)

Data added on top of v8 (all public or made locally; no subscription tokens):

| source | texts | aimed at |
|---|---|---|
| benchmark *training* splits (`train/build_public_train.py`): DetectRL (ChatGPT, Claude-instant, PaLM, Llama-2 on arXiv / XSum / stories / Yelp, plus back-translation, DIPPER, polish, typo and evasion-prompt attacks — and the same attacks on human text), Ghostbuster (non-test files), OUTFOX train | 20,296 | 2023 Claude-instant / PaLM text, back-translation |
| local back-translation of modern AI *and* human messages through German / French with open MarianMT models (`train/build_backtrans.py`) | 2,399 | the humanizer trick we caught 13–33 % of |
| short real human messages: Reddit comments 2005–19 (GoEmotions), SAMSum chats, EmpatheticDialogues (`train/build_short_human2.py`) | 4,494 | humans wrongly flagged on short text |

The benchmarks' test texts were excluded by hash (167 dropped). Training: 10,996 steps, 106 min on an A40, $1.04.

**Result: worse on its own.** On our held-out test sets v9 scored 92.8 % on the 291 texts ZeroGPT also judged
(v8: 96.6 %), caught 79 % of humanized AI (v8: 89 %) and flagged 8.3 % of native-English human texts (v8: 6.0 %).
Likely causes, most suspected first:
1. **The old benchmark data swamped the mix.** ~20k 2023 texts became about a quarter of the extra data, so the model
   spent much of its training on old models and long-form domains instead of modern, casual writing.
2. **Translationese is in both labels on purpose** (back-translated human = human, back-translated AI = AI). That
   is right, but it removes an easy cue the model leaned on for disguised AI — consistent with the humanized drop.
3. **"AI-polished human = AI-edited"** (~800 texts) teaches suspicion of polish — consistent with more native-English
   human texts flagged.
4. **Checkpoint selection** ran on a dev set now dominated by benchmark text.
5. Run-to-run noise (a point or two, not ten).

**Inside a combiner (finished 2026-11-12 evening).** The same length-aware, casual-aware fit as the live engine
(`eval/stack_casual.py fit --length`, out-of-fold), scored on all four yardsticks:

| | our test sets (vs ZeroGPT 280/291) | humanized AI | 400 review texts (ZeroGPT 9.9) | owner's genres / short AI (review round 3) | public, 510 texts ZeroGPT judged (ZeroGPT 96.9 %) | back-translated *modern* AI, held out (142) | short human messages flagged (426) |
|---|---|---|---|---|---|---|---|
| **v0 + v8 (live)** | **285** | **91 %** | **9.8** | **56/57 · 20/21** | 84.3 % | 107 | 4 |
| v0 + v9 | 276 | 83 % | 9.7 | 54/57 · 18/21 | **95.9 %** | **110** | **1** |
| v0 + v8 + v9 | 284 | 90 % | 9.8 | 55/57 · 19/21 | 88.0 % | 107 | 2 |

- **v9 closed the public-benchmark gap**, but that data is in-distribution for it: v9 trained on the same
  benchmarks' training splits (same generators, domains and attacks; the test texts themselves held out by hash).
  DetectRL back-translation on the texts ZeroGPT judged: live 11/48, v0 + v9 41/48, ZeroGPT 41/48; Claude-instant
  4/12 → 11/12 (ZeroGPT 12/12).
- **On modern back-translated messages the gap was never large.** On v9's held-out split (German / French round
  trips of 2025–26 AI messages, which no model trained on) the live engine already catches 107/142 (75 %), v9 110.
  The 13–33 % weakness was mostly about *2023* models' text, not the trick itself.
- **Where it matters for the owner, v9 is a little worse:** fewer AI messages caught in the owner's genres and on
  short texts, humanized AI down 8 points, 9 fewer right on our test sets. Its one real gain there is fewer short
  human messages flagged (1 vs 4 of 426).

**Decision: the live engine stays v0 + v8.** The v9 combiners are saved locally (`models/stacker-v0-mb9-len.json`,
`models/stacker-v0-mb8-mb9-len.json`, not deployed).

Next, if resumed — v10: the benchmark data cut to ~8k texts so it can't swamp the modern data, no
"polished human = AI-edited" label, back-translation kept in both labels, one pass over the data instead of two
(half the time and cost), and a faster GPU when one is available; then the same yardsticks — and judge the public benchmarks with the benchmark-trained
models' in-distribution advantage in mind.

_Paused 2026-11-12 15:30. Runpod spend so far: $4.46 of $10._
