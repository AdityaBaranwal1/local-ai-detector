# Round 2 preregistered analysis

Written before detector calls. Freeze all 120+ texts and SHA-256/timestamp manifest before either detector is run. No outcome-based changes. Fresh texts must not match round 1. Store all raw responses and failures; do not count unavailable checks as correct.

Thresholds: ours /api/detect ai_score >= 0.5; ZeroGPT fakePercentage >= 50. Separately report /verify FAIL (including its stricter threshold and any tell veto).

Primary score, identical to round 1 and applied identically to both detectors:
10 * (0.40 * human specificity + 0.20 * raw-AI sensitivity + 0.40 * casual/humanized-AI sensitivity).
AI-polished human is reported separately and excluded from this diagnostic score. Accuracy includes it as an AI-derived positive, with a second accuracy excluding polishing. Report exact numerators and denominators.

Fresh set is primary; round-1 set is a retrospective comparison, not unseen validation. Rerun all three endpoints, including ZeroGPT, on round 1 as well. Do not reuse old scores as current measurements.

Parity rule: ours must have at least ZeroGPT's polishing-excluded accuracy, no higher aggregate human FPR, and no lower hard-AI sensitivity. Frontier judgment additionally requires no clearly elevated human subgroup FPR and high sensitivity across current models and hard styles. Small samples and source clustering limit all conclusions; do not claim universal superiority or label a disputed polishing case as proof of fully AI authorship.

Human bias strata: native/proficient proxy vs documented non-native, formal vs casual, short <=150 words vs long >150 words, and domain/genre. Report subgroup counts and acknowledge overlap, unknown L1, historical prose and exam-task proxies. Inspect raw errors, engine/version strings, complete statuses, and text hashes before writing the report.
