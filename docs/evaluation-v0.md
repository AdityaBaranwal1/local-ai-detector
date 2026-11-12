# Evaluation of the starting model (v0, 2026-11-11)

_Snapshot of the first model (tropa-mini at its published revision, no fine-tuning). Later versions: [evaluation.md](evaluation.md)._


743 texts: 554 human (pre-2020 news, Enron email, reddit posts/stories, and Liang et al. 2023's essay sets), 66 fresh
AI texts from 8 current models (Claude Opus/Sonnet/Haiku 5.5, Fable 5.2, Sonnet 4.5, GPT-6 Sol, GPT-5.6 Terra,
Gemini 3 Flash; 12 genres incl. cover letters, LinkedIn, reddit, "casual"), 32 of those rewritten by an LLM "to sound
human", 91 human essays polished by GPT-4. Full output: `eval/RESULTS.txt`.

| flagged as AI at the 50 % line | this detector |
|---|---|
| human: CS224N papers / US college essays / US student essays | 1 / 4 / 0 % |
| human: news / Enron email / reddit posts / reddit stories | 2 / 2 / 0 / 8 % |
| human: TOEFL essays, non-native | **33 %** |
| fresh AI text, 8 models | **98.5 %** (100 % for 7 of 8; Opus 5.5 92 %) · AUROC 0.994 |
| AI text rewritten by an LLM "to sound human" | 72 % |
| human essays polished by GPT-4 | 90 % |

Other engines tried (AUROC, human vs raw AI): desklib 0.890 · fakespot 0.764 · TMR 0.479 · Binoculars (Qwen2.5-1.5B
pair) 0.597 · Fast-DetectGPT 0.557; a learned combination of them all did worse than tropa-mini alone.

### Against ZeroGPT (zerogpt.com today, through its web API; `eval/sites/`)

Stratified sample of 117 of the texts above (`eval/zerogpt_compare.py` → `sites/zerogpt_sample.json`):

| flagged as AI (≥ 50 %) | n | ZeroGPT | this detector |
|---|---|---|---|
| human, native English (essays, email, news, reddit) | 45 | 0 % | 0 % |
| human, non-native (TOEFL) | 12 | 0 % | 33 % |
| raw AI, 8 current models | 36 | 100 % | 100 % |
| AI rewritten by an LLM to sound human | 16 | **100 %** | 75 % |
| human essays polished by GPT-4 | 8 | 100 % | 100 % |

And 12 texts shaped like the real use (`sites/results.json`): 8 written by Claude Haiku 5.5 (3 emails, 3 application
text-box answers, 2 "typed casually"), 4 human (2 Enron emails, 2 pre-2023 college essays): **ZeroGPT 12/12, this
detector 10/12** — it missed an all-lowercase "ok so…" answer (2 %) and scored one real college essay 100 %.

So ZeroGPT (2026) is the stronger detector on humanized and non-native text; this one matches it on plain human and
plain AI text, runs offline, and says which sentences drive the score. Caveat: the human corpora are public, so
ZeroGPT may have trained on them. Only ZeroGPT could be tested automatically: GPTZero and Sapling need an account/API
key, QuillBot and Undetectable sit behind a "verify you are human" check.

## Speed (eval/bench.py → eval/BENCH.txt, 2026-11-11)

| words | GPU | CPU (a render holds the GPU) |
|---|---|---|
| 100 | 0.6 s | 0.9 s |
| 250 | 2.4 s | 4.0 s |
| 500 | 8.4 s | 14.7 s |
| 1000 | 20 s | 38 s |
| 2000 | 44 s | 87 s |

Short texts (≤ 225 words) take one extra pass per sentence for the highlights: ~1–4 s on the GPU (measured before this
mode existed: the table above). Cold start 4.1 s. RAM 3.1 GiB while loaded, VRAM 1.5 GiB while it holds the GPU.
Checks are served one at a time (4 × 250 words at once: 9.7 s). A render that asks for the GPU gets it after
0.1–1.4 s when the detector is idle, or when the check in flight finishes (≤ 44 s for 2000 words).
