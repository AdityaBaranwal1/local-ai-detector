# Independent review brief: local AI-text detector vs ZeroGPT

You are reviewing a local AI-text detector against ZeroGPT, independently. The builder (another agent) will
improve the detector between rounds, and will **not** train on any text you create or collect: your test set
stays unseen. Your job is a fair, hard, honest test and a score.

## What to test

- **Ours:** a service on this machine. `POST http://127.0.0.1:8323/api/detect` with JSON `{"text": "..."}` returns
  `ai_score` (0–1). Count it as "flagged AI" when `ai_score >= 0.5`. Also `POST http://127.0.0.1:8323/verify`
  (same body) returns `verdict` PASS/FAIL: the stricter gate agents use before text goes out under the owner's
  name; report it too. Usage text: `GET http://127.0.0.1:8323/verify`. One request at a time.
- **ZeroGPT:** `POST https://api.zerogpt.com/api/detect/detectText` with JSON `{"input_text": "..."}` and headers
  `Origin: https://www.zerogpt.com`, `Referer: https://www.zerogpt.com/`, a normal browser User-Agent. The answer's
  `data.fakePercentage` (0–100) is its AI %; count ≥ 50 as flagged. Wait ≥ 5 s between requests; retry politely on
  errors. Do not send anything personal: only texts you generated or public pre-2022 human texts.
- Other public detectors only if they work without logins, keys, or getting past "verify you are human" checks
  (don't automate past those).

## The test set (build it yourself, fresh each round)

At least **120 texts, 60–500 words each**, roughly half human and half AI, covering what the owner actually checks
— **emails** (cold outreach, follow-up, reference request, work email), **job-application form answers**, **cover
letters**, **LinkedIn posts / short messages** — plus essays, reviews, forum posts and short stories.

- **Human** must be genuinely human and pre-2022 (before ChatGPT). Include a good share of **non-native English**
  writers and informal/casual writing. Use public sources the builder has **not** used. Already used (avoid):
  Jabarian & Imas DetectionAI, ELLIPSE / Feedback Prize ELL, ICNALE, DAIGT / PERSUADE, RAID, AESLC (Enron),
  webis/tldr-17, euclaise/writingprompts, CNN/DailyMail, Liang et al. ChatGPT-Detector-Bias (TOEFL, college,
  Hewlett, CS224N), textsightai 2026 benchmark. Note each text's source.
- **AI** from several current models: write some yourself, and get others through the OpenAI-compatible proxy on
  this box (`http://127.0.0.1:8317/v1/chat/completions`; the API key is in `~/docker/cliproxyapi/config.yaml` under
  `api-keys`; models e.g. claude-haiku-5-5, claude-sonnet-5-5, gemini-3-flash, gpt-5.6-luna, gpt-6-luna,
  gpt-oss-120b-medium — keep usage modest, cheap models first). Include hard cases: AI told to sound casual or
  human, all-lowercase texting style, AI imitating a non-native writer, AI text run through a "humanize/paraphrase"
  rewrite, and human text "polished" by an AI (label that one AI-edited and report it separately).
- Save the set with labels and sources (`texts.json`), and both detectors' raw answers (`results.json`).

## Report

Write `REPORT.md` in your round folder (the builder gives you its path): method, the per-category table (human
native / human non-native / AI raw / AI casual-or-humanized / AI-polished human) with flagged-as-AI rates for ours
and ZeroGPT, overall accuracy for each, the 5–10 most instructive failures of ours (short quotes are fine), and:

- **A score from 0 to 10 for our detector's AI-detection quality**, with ZeroGPT's performance on the same texts as
  the reference point, and one paragraph saying why.
- **One line: is ours at least as good as ZeroGPT on this set? (yes/no)**
- What would most improve ours.

Be strict and specific. Don't edit the detector's code or its service.
