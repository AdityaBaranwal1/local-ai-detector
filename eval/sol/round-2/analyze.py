import pathlib,json,hashlib,collections,math
R=pathlib.Path(__file__).resolve().parent
CATS=['human_native','human_non_native','ai_raw','ai_casual_or_humanized','ai_polished_human']
def analyze(tp,rp):
 ts=json.loads(tp.read_text());rr=json.loads(rp.read_text());assert rr.get('finished_at');assert hashlib.sha256(tp.read_bytes()).hexdigest()==rr['texts_file_sha256'];by={x['id']:x for x in ts};rows=rr['rows'];assert len(rows)==len(ts)==120 and len({x['id'] for x in rows})==120
 for r in rows:
  assert r['sha256']==hashlib.sha256(by[r['id']]['text'].encode()).hexdigest()
  for k in ['ours','verify','zerogpt']:assert r[k]['ok'],(r['id'],k,r[k])
 def flag(r,k):
  raw=r[k]['raw'];return raw['ai_score']>=.5 if k=='ours' else raw['data']['fakePercentage']>=50 if k=='zerogpt' else raw['verdict']=='FAIL'
 def stats(sub):
  d={'n':len(sub)}
  for k in ['ours','zerogpt','verify']:
   fp=sum(r['label']=='human' and flag(r,k) for r in sub);fn=sum(r['label']!='human' and not flag(r,k) for r in sub);tp=sum(r['label']!='human' and flag(r,k) for r in sub);tn=sum(r['label']=='human' and not flag(r,k) for r in sub)
   d[k]=dict(flagged=sum(flag(r,k) for r in sub),fp=fp,fn=fn,tp=tp,tn=tn,correct=tn+tp,accuracy=(tn+tp)/len(sub) if sub else None)
  return d
 cats={c:stats([r for r in rows if r['category']==c]) for c in CATS}
 quality={k:10*(.4*(1-cats['human_native'][k]['flagged']/60-cats['human_non_native'][k]['flagged']/60)+.2*cats['ai_raw'][k]['flagged']/20+.4*cats['ai_casual_or_humanized'][k]['flagged']/30) for k in ['ours','zerogpt','verify']}
 humans=[r for r in rows if r['label']=='human'];bias={}
 for name,pred in [('native_or_proficient',lambda t:t['category']=='human_native'),('non_native',lambda t:t['category']=='human_non_native'),('formal',lambda t:t.get('register', 'casual' if t['genre'] in ['forum_work_or_application','fiction','hobbies'] else 'formal')=='formal'),('casual',lambda t:t.get('register','casual' if t['genre'] in ['forum_work_or_application','fiction','hobbies'] else 'formal')=='casual'),('short_le150',lambda t:t['word_count']<=150),('long_gt150',lambda t:t['word_count']>150)]:
  bias[name]=stats([r for r in humans if pred(by[r['id']])])
 for g in sorted({r['genre'] for r in humans}):bias['domain:'+g]=stats([r for r in humans if r['genre']==g])
 for source in sorted({by[r['id']]['source']['collection'] for r in humans}):bias['source:'+source]=stats([r for r in humans if by[r['id']]['source']['collection']==source])
 models={m:stats([r for r in rows if r.get('model')==m and r['category']!='ai_polished_human']) for m in sorted({r['model'] for r in rows if r.get('model')})}
 subtypes={s:stats([r for r in rows if r['category']=='ai_casual_or_humanized' and by[r['id']].get('subtype')==s]) for s in sorted({by[r['id']].get('subtype') for r in rows if r['category']=='ai_casual_or_humanized'})}
 versions=collections.Counter(json.dumps(r['verify']['raw'].get('versions',{}),sort_keys=True) for r in rows)
 devices=collections.Counter(r['ours']['raw'].get('device','not reported') for r in rows)
 store_hits=collections.Counter(str(r['verify']['raw'].get('from_store')) for r in rows)
 errors=[]
 for r in rows:
  if flag(r,'ours')!=(r['label']!='human') or flag(r,'verify')!=(r['label']!='human'):
   t=by[r['id']];errors.append(dict(id=r['id'],category=r['category'],genre=r['genre'],text=t['text'],source=t['source'],ours=r['ours']['raw']['ai_score'],zero=r['zerogpt']['raw']['data']['fakePercentage'],verify=r['verify']['raw'],detect=r['ours']['raw']))
 nonpol=[r for r in rows if r['category']!='ai_polished_human'];ours_only=zero_only=0
 for r in nonpol:
  a=flag(r,'ours')==(r['label']!='human');b=flag(r,'zerogpt')==(r['label']!='human');ours_only+=a and not b;zero_only+=b and not a
 n=ours_only+zero_only;p=min(1,2*sum(math.comb(n,i) for i in range(min(ours_only,zero_only)+1))/2**n) if n else 1
 mismatch=[r['id'] for r in rows if abs(r['ours']['raw']['ai_score']-r['verify']['raw'].get('ai_score',r['ours']['raw']['ai_score']))>1e-6]
 return dict(categories=cats,quality=quality,all=stats(rows),excluding_polish=stats(nonpol),bias=bias,models=models,hard_subtypes=subtypes,verify_versions=dict(versions),devices=dict(devices),verify_store_hits=dict(store_hits),ours_only_correct=ours_only,zero_only_correct=zero_only,mcnemar_exact_p=p,detect_verify_score_mismatches=mismatch,errors=errors,started_at=rr['started_at'],finished_at=rr['finished_at'],health_before=rr.get('health_before'),health_after=rr.get('health_after'),texts_file_sha256=rr['texts_file_sha256'],word_range=[min(t['word_count'] for t in ts),max(t['word_count'] for t in ts)])
fresh=analyze(R/'texts.json',R/'results.json');old=analyze(R.parent/'round-1/texts.json',R/'round1-results.json');out=dict(fresh=fresh,round1_rerun=old);(R/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
for name,s in out.items():
 print('\nDATASET',name,'QUALITY',s['quality'],'VERSIONS',s['verify_versions']);print('ALL',s['all']);print('EXCLUDING POLISH',s['excluding_polish']);print('CATEGORIES',s['categories']);print('BIAS',json.dumps(s['bias']));print('HARD',s['hard_subtypes']);print('MODELS',s['models']);print('ERRORS',[(e['id'],round(e['ours']*100,2),e['zero'],e['verify']['verdict'],e['text'][:100]) for e in s['errors']])
