import pathlib,json
R=pathlib.Path(__file__).resolve().parent
S=json.loads((R/'phase_b_summary.json').read_text())
NAMES={'human_native':'Human native/proficient proxy','human_non_native':'Human non-native','ai_raw':'AI raw','ai_casual_or_humanized':'AI casual/humanized','ai_polished_human':'AI-polished human'}
def frac(s,k):
 n=s['n'];return f"{s[k]['flagged']}/{n} — {100*s[k]['flagged']/n:.1f}%" if n else '— (n=0)'
def table(title,items):
 lines=['## '+title,'','| Group | N | Ours ≥50% | ZeroGPT ≥50% | `/verify` FAIL |','|---|---:|---:|---:|---:|']
 for name,s in items:lines.append(f"| {name} | {s['n']} | {frac(s,'ours')} | {frac(s,'zerogpt')} | {frac(s,'verify')} |")
 return '\n'.join(lines)+'\n\n'
lines=[]
for key,title in [('round3','Round 3 — primary'),('round2_rerun','Round 2 rerun — development-informed regression set')]:
 s=S[key];lines+=['# '+title,'',f"Fixed score: ours **{s['quality']['ours']:.4f}/10**, ZeroGPT **{s['quality']['zerogpt']:.4f}/10**.",'']
 lines.append(table('Per-category results',[(NAMES[c],v) for c,v in s['categories'].items()]))
 lines+=['## Accuracy','','| Convention | Detector/gate | Correct / N | Accuracy | TP | TN | FP | FN |','|---|---|---:|---:|---:|---:|---:|---:|']
 for field,label in [('excluding_polish','Excluding polishing'),('all','Polishing counted AI-derived positive')]:
  a=s[field]
  for k in ['ours','zerogpt','verify']:
   x=a[k];lines.append(f"| {label} | {k} | {x['correct']}/{a['n']} | {100*x['accuracy']:.2f}% | {x['tp']} | {x['tn']} | {x['fp']} | {x['fn']} |")
 lines+=['','']
 lines.append(table('Priority slice',[(name,s['priority'][name]) for name in ['human','generated','raw','hard','polished']]))
 lines.append(table('Priority genres — human false positives',[(g,v['human']) for g,v in s['priority_genres'].items()]))
 lines.append(table('Priority genres — generated-AI sensitivity',[(g,v['generated']) for g,v in s['priority_genres'].items()]))
 lines.append(table('Priority genres — AI-polished human',[(g,v['polished']) for g,v in s['priority_genres'].items()]))
 lines.append(table('30–60-word slice',[(name,s['short'][name]) for name in ['human','generated','raw','hard','polished']]))
 lines.append(table('Bias — human false positives (overlapping groups)',list(s['bias'].items())))
 lines.append(table('Hard AI styles',list(s['hard_subtypes'].items())))
 lines.append(table('Generation model — polishing excluded',list(s['models'].items())))
 lines+=['## Local errors (full text remains in frozen set)','','| ID | Category / genre | Ours AI % | ZeroGPT AI % | Gate | Short quote |','|---|---|---:|---:|---|---|']
 for e in s['errors']:
  q=' '.join(e['text'].split())[:150].replace('|','\\|');lines.append(f"| {e['id']} | {e['category']} / {e['genre']} | {100*e['ours']:.2f} | {e['zero']:.2f} | {e['verify']['verdict']} | {q}… |")
 lines+=['','']
(R/'phase_b_tables.md').write_text('\n'.join(lines))
print('Wrote phase_b_tables.md')
