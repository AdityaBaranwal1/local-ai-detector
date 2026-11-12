import pathlib,json,hashlib,datetime,collections,math
R=pathlib.Path(__file__).resolve().parent
T=json.loads((R/'texts.json').read_text());M=json.loads((R/'manifest.json').read_text());Z=json.loads((R/'zerogpt.json').read_text())
H=hashlib.sha256((R/'texts.json').read_bytes()).hexdigest();assert H==M['texts_file_sha256']==Z['texts_file_sha256']
assert len(T)==160 and len({x['id'] for x in T})==160
byid={x['id']:x for x in T};rows={r['id']:r for r in Z['rows']};assert len(rows)==len(Z['rows'])
assert Z.get('complete') is True and len(rows)==len(T) and all(r['ok'] for r in rows.values()),'Do not issue a completion report for an incomplete run'
for x in T:assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256']
allattempts=[]
for r in Z['rows']:
 x=byid[r['id']];assert r['sha256']==x['sha256']
 for a in r['attempts']:
  assert a['started_at']>M['frozen_at'];allattempts.append(a)
 if r['ok']:
  score=float(r['raw']['data']['fakePercentage']);assert r['score_percent']==score and r['flagged_ai']==(score>=50)
  last=r['attempts'][-1];assert json.loads(last['raw_http_body'])==r['raw']
allattempts.sort(key=lambda a:a['started_at'])
gaps=[]
for a,b in zip(allattempts,allattempts[1:]):
 gap=(datetime.datetime.fromisoformat(b['started_at'])-datetime.datetime.fromisoformat(a['finished_at'])).total_seconds();gaps.append(gap);assert gap>=5.29,gap
cats=['human_native','human_non_native','ai_raw','ai_casual_or_humanized','ai_polished_human']
def rate(ids):
 complete=[rows[i] for i in ids if i in rows and rows[i]['ok']];flags=sum(r['flagged_ai'] for r in complete)
 return dict(n=len(ids),completed=len(complete),flagged=flags,flag_rate=flags/len(complete) if complete else None)
summary=dict(round=3,phase='A',frozen_at=M['frozen_at'],texts_file_sha256=H,completed=sum(r['ok'] for r in Z['rows']),sample_count=len(T),word_range=[min(x['word_count'] for x in T),max(x['word_count'] for x in T)],local_service_called=False,zerogpt_threshold_percent=50,by_category={c:rate([x['id'] for x in T if x['category']==c]) for c in cats},short_30_60={c:rate([x['id'] for x in T if x['category']==c and x['short_30_60']]) for c in cats},priority_genres={},models={},human_sources=dict(collections.Counter(x['source']['collection'] for x in T if x['label']=='human')),human_L1=dict(collections.Counter(x['source']['native_language'] for x in T if x['category']=='human_non_native')),hard_styles=dict(collections.Counter(x['subtype'] for x in T if x['category']=='ai_casual_or_humanized')),priority_count=sum(x['priority'] for x in T),short_count=sum(x['short_30_60'] for x in T),request_attempts=len(allattempts),retry_attempts=len(allattempts)-len(Z['rows']),minimum_completion_to_next_start_seconds=min(gaps) if gaps else None,finished_at=Z.get('finished_at'))
for g in ['linkedin_outreach','linkedin_post','recruiter_reply','networking_referral_ask','follow_up','thank_you_note','application_answer']:
 xs=[x for x in T if x['priority_genre']==g];summary['priority_genres'][g]=dict(total=len(xs),human=sum(x['label']=='human' for x in xs),generated=sum(x['label']=='ai' for x in xs),polished=sum(x['label']=='ai_edited' for x in xs),short_30_60=sum(x['short_30_60'] for x in xs),zerogpt=rate([x['id'] for x in xs]))
for model in sorted({x['source'].get('model') for x in T if x['label']!='human'}):summary['models'][model]=dict(total=sum(x['source'].get('model')==model for x in T),by_category=dict(collections.Counter(x['category'] for x in T if x['source'].get('model')==model)))
# Keep the actual number of successful/failed generation HTTP attempts auditable.
gen=[]
for p in (R/'generation').glob('*-attempt-*.json'):
 j=json.loads(p.read_text());gen.append(dict(file=p.name,model=j['request']['model'],http_status=j['http_status'],started_at=j['started_at']))
errors=[]
for p in (R/'generation').glob('*-error-*.json'):
 j=json.loads(p.read_text());errors.append(dict(file=p.name,**j))
summary['generation_successful_http_attempts']=len(gen);summary['generation_error_records']=len(errors)
summary['generation_failed_http_attempts']=sum('http_status' in x for x in errors)
(R/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
percent=lambda r: 'unavailable' if r['flag_rate'] is None else f"{100*r['flag_rate']:.1f}%"
lines=['# Round 3 — phase A complete','',f"Frozen at **{M['frozen_at']}**, before any detector request. **No local detector endpoint was called** (including health). No detector/model/service edits, commits or training were performed.",'',f"**ZeroGPT: {summary['completed']}/{len(T)} completed.** Flagged means `data.fakePercentage >= 50`. Raw response bodies, parsed responses and all attempt timestamps are in `zerogpt.json`.",'','## Frozen set','',f"{len(T)} texts, {summary['word_range'][0]}–{summary['word_range'][1]} whitespace-separated words. 80 human, 65 wholly generated, 15 AI-polished human. {summary['priority_count']} priority-genre texts; {summary['short_count']} in the inclusive 30–60-word slice.",'','| Category | N | ZeroGPT flags | Flag rate | 30–60 words: N | Short flags | Short flag rate |','|---|---:|---:|---:|---:|---:|---:|']
for c in cats:
 r=summary['by_category'][c];s=summary['short_30_60'][c];lines.append(f"| {c} | {r['n']} | {r['flagged']}/{r['completed']} | {percent(r)} | {s['n']} | {s['flagged']}/{s['completed']} | {percent(s)} |")
lines+=['','Human flag rates are false-positive rates; generated-AI flag rates are sensitivity. Polishing is mixed provenance and is reported separately. No local quality score, parity judgment or frontier verdict belongs in phase A.','','## Priority genres','','| Genre | Human | Generated AI | AI-polished | Total | 30–60 words |','|---|---:|---:|---:|---:|---:|']
for g,v in summary['priority_genres'].items():lines.append(f"| {g} | {v['human']} | {v['generated']} | {v['polished']} | {v['total']} | {v['short_30_60']} |")
lines+=['','## Human sources','',
'- **40 PELIC v1.1 learner texts**, collected 2005–2012, published in 2020: [official dataset](https://github.com/ELI-Data-Mining-Group/PELIC-dataset), [release DOI](https://doi.org/10.5281/zenodo.3991977). Forty different anonymous authors with recorded L1 metadata. Seven application answers, five thank-you notes, ten other short compositions and eighteen longer compositions. Four thank-you notes use contiguous bodies to omit publisher-anonymized salutations/signatures; wording was not corrected. Original answer IDs, creation dates, prompts, L1s, hashes and excerpt offsets are retained.',
'- **12 authentic historical letters by Robert Louis Stevenson**, published in *Letters*, volume 1 (1899; author died in 1894): [Project Gutenberg 622](https://www.gutenberg.org/ebooks/622). Includes referral/academic-position-support requests, professional thanks and follow-ups. These are historical genre analogies, **not modern LinkedIn or recruiter communications**.',
'- **6 T. H. Huxley scientific/philosophical excerpts**, *Darwiniana*, collected essays volume II (1893): [Project Gutenberg 6919](https://www.gutenberg.org/ebooks/6919).',
'- **6 H. G. Wells narrative excerpts**, *The Time Machine* (1895): [Project Gutenberg 35](https://www.gutenberg.org/ebooks/35).',
'- **16 practical forum questions**, eight each from Bicycles and Home Improvement Stack Exchange, created before 2018, with no post-2021 edits in acquired metadata. URLs/IDs/timestamps are in `texts.json`. These two sites are outside the listed builder/earlier-round exclusions; individual first languages are unknown, so they are native/proficient **proxies**, not verified native speakers.',
'', 'Recorded non-native first-language counts: '+', '.join(f'{k} {v}' for k,v in summary['human_L1'].items())+'.',
'', 'Source novelty: no human corpus/book/author or Stack Exchange site from rounds 1–2 is reused, although broad redistribution hosts (GitHub/Gutenberg/Stack Exchange) overlap. No listed builder training collection is intentionally used. This cannot establish absence from detector pretraining. NUCLE was considered but **not used**, because its license restricts third-party transfer; only its public README/license were downloaded. Unused Muse/UCSF source downloads are not part of the evaluation.',
'', '## AI sources','']
for model,v in summary['models'].items():lines.append(f"- `{model}`: **{v['total']} texts**; "+', '.join(f'{c} {n}' for c,n in v['by_category'].items())+'.')
lines+=['',f"Hard-style composition: "+', '.join(f'{k} {v}' for k,v in summary['hard_styles'].items())+'. Natural restrained, concrete messages are a substantial prespecified part of the hard group, not a post-outcome selection.',
'', 'Proxy prompts, raw HTTP replies, rejected/unused generation attempts and successful batches are saved under `generation/`. Model IDs are requested provenance, not independent authentication of the underlying model or account. Grok returned 402/503 and older Haiku returned 404/503, yielding no included samples; Muse was not called. Sonnet’s first valid raw batch exceeded some requested short lengths. It was retained untruncated, with the **actual** short slice assigned by word count; a redundant second raw batch is preserved but unused. Gemini’s original polishing batch substituted the fictional profile for the originals; it was excluded before testing and replaced with a source-only editing batch whose facts were checked against the originals. Before freezing, publisher-anonymized signatures were omitted from two human thank-you bodies and the corresponding Gemini-polished signature; one Sonnet placeholder was replaced with a fictional addressee, documented as a reviewer post-edit. No frozen text was changed. All such decisions preceded detector calls. No Fable or Opus calls were made.',
'',f"Generation HTTP accounting: {summary['generation_successful_http_attempts']} successful HTTP replies, {summary['generation_failed_http_attempts']} failed HTTP attempts; model-output validation error records are separate from HTTP failures. Twelve selected successful batch files supply the 60 proxy texts. Reviewer writing is not a proxy call.",
'', '## Integrity and request pacing','',f"SHA-256 of the exact `texts.json` bytes:\n\n`{H}`",'',M['deduplication'],
'',f"ZeroGPT request attempts: {summary['request_attempts']}; retries beyond the first per text: {summary['retry_attempts']}. Minimum observed preceding-completion to next-start gap: {summary['minimum_completion_to_next_start_seconds']:.3f} seconds. All request start timestamps follow the freeze. Completion: {summary['finished_at']}.",
'', '## Limits and handoff','',
'- This tests preselected, source-clustered samples, not a representative population. Forty learner texts share PELIC; historical authors and repeated synthetic profiles introduce further clustering.',
'- Human job-search coverage remains a limitation: pedagogical application answers and thanks plus historical correspondence, **no verified authentic modern human LinkedIn posts/outreach or recruiter replies**. Priority totals include labeled historical/educational proxies. Do not claim that a strong aggregate result validates the full modern use case.',
'- Some short human compositions are incomplete classroom drafts. The 30–60-word group is deliberately retained and separately measured; short-text certainty should not be inferred from its raw percentages.',
'- AI-polished examples share human parents, and some humanized AI share raw parents. They are not independent observations. Polishing is excluded from the fixed primary quality formula.',
'- Phase B should read the existing frozen files, verify the hash, then evaluate the newly deployed local service. Earlier rounds can be rerun locally; their saved ZeroGPT responses should be reused. Do not regenerate or refreeze this round.',
'', 'Files: `texts.json`, `manifest.json`, `zerogpt.json`, `summary.json`, `PHASE_A.md`, `ANALYSIS_PLAN.md`, generation/acquisition logs and scripts. `texts.json` and `manifest.json` are read-only to reduce accidental edits. This phase intentionally does not create a final comparative `REPORT.md`.']
(R/'PHASE_A.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
