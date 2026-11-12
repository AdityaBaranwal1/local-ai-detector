import pathlib,json,hashlib,collections,math,datetime
R=pathlib.Path(__file__).resolve().parent
CATS=['human_native','human_non_native','ai_raw','ai_casual_or_humanized','ai_polished_human']
KEYS=['ours','zerogpt','verify']
def flag(r,k):
 raw=r[k]['raw'];return raw['ai_score']>=.5 if k=='ours' else raw['data']['fakePercentage']>=50 if k=='zerogpt' else raw['verdict']=='FAIL'
def stats(rows):
 d={'n':len(rows)}
 for k in KEYS:
  fp=sum(r['label']=='human' and flag(r,k) for r in rows);fn=sum(r['label']!='human' and not flag(r,k) for r in rows);tp=sum(r['label']!='human' and flag(r,k) for r in rows);tn=sum(r['label']=='human' and not flag(r,k) for r in rows)
  d[k]=dict(flagged=sum(flag(r,k) for r in rows),fp=fp,fn=fn,tp=tp,tn=tn,correct=tn+tp,accuracy=(tn+tp)/len(rows) if rows else None)
 return d
def wilson(k,n):
 if not n:return None
 z=1.95996398454;p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [max(0,center-half),min(1,center+half)]
def analyze(n,filename):
 folder=R.parent/f'round-{n}';tp=folder/'texts.json';ts=json.loads(tp.read_text());out=json.loads((R/filename).read_text());manifest=json.loads((folder/'manifest.json').read_text())
 assert out['complete'] and out['finished_at'];assert hashlib.sha256(tp.read_bytes()).hexdigest()==out['texts_file_sha256']==manifest['texts_file_sha256']
 rows=out['rows'];by={t['id']:t for t in ts};assert len(rows)==len(ts)==len({r['id'] for r in rows})
 oldfile=folder/('zerogpt.json' if n==3 else 'results.json');old=json.loads(oldfile.read_text());oldby={r['id']:r for r in old['rows']}
 attempts=[]
 for r in rows:
  t=by[r['id']];assert r['sha256']==t['sha256']==hashlib.sha256(t['text'].encode()).hexdigest()
  assert r['zerogpt']==(oldby[r['id']] if n==3 else oldby[r['id']]['zerogpt'])
  for k in KEYS:assert r[k]['ok'],(r['id'],k)
  assert r['verify']['raw']['complete']
  for k in ['ours','verify']:
   for a in r[k]['attempts']:
    assert datetime.datetime.fromisoformat(a['started_at'])>datetime.datetime.fromisoformat(manifest['frozen_at']);attempts.append(a)
   assert json.loads(r[k]['attempts'][-1]['raw_http_body'])==r[k]['raw']
 attempts.sort(key=lambda a:a['started_at'])
 for a,b in zip(attempts,attempts[1:]):assert a['finished_at']<=b['started_at'],'Overlapping local requests'
 cats={c:stats([r for r in rows if r['category']==c]) for c in CATS};humans=[r for r in rows if r['label']=='human'];gen=[r for r in rows if r['label']=='ai'];pol=[r for r in rows if r['label']=='ai_edited'];nonpol=[r for r in rows if r['category']!='ai_polished_human']
 quality={k:10*(.4*(1-stats(humans)[k]['flagged']/len(humans))+.2*cats['ai_raw'][k]['flagged']/cats['ai_raw']['n']+.4*cats['ai_casual_or_humanized'][k]['flagged']/cats['ai_casual_or_humanized']['n']) for k in ['ours','zerogpt']}
 priority_r2={'cold_outreach','followup_email','work_email','reference_request','linkedin_post','linkedin_message','application_answer','cover_letter'}
 priority=lambda t:t.get('priority',False) if n==3 else t['genre'] in priority_r2
 bias={}
 def group(name,p):bias[name]=stats([r for r in humans if p(by[r['id']])])
 for name,p in [('native_or_proficient',lambda t:t['category']=='human_native'),('non_native',lambda t:t['category']=='human_non_native'),('formal',lambda t:t.get('formality',t.get('register'))=='formal'),('casual',lambda t:t.get('formality',t.get('register'))=='casual'),('casual_professional',lambda t:t.get('formality',t.get('register'))=='casual_professional'),('short_30_60',lambda t:30<=t['word_count']<=60),('longer_61plus',lambda t:t['word_count']>60),('short_le150',lambda t:t['word_count']<=150),('long_gt150',lambda t:t['word_count']>150),('priority_human',priority)]:group(name,p)
 for field,prefix in [('genre','domain'),('collection','source'),('native_language','L1')]:
  get=lambda t:t.get(field) if field=='genre' else t['source'].get(field)
  for v in sorted({get(by[r['id']]) for r in humans}-{None}):group(prefix+':'+v,lambda t,v=v:get(t)==v)
 priority_rows=[r for r in rows if priority(by[r['id']])]
 def split(sub):return dict(all=stats(sub),human=stats([r for r in sub if r['label']=='human']),generated=stats([r for r in sub if r['label']=='ai']),raw=stats([r for r in sub if r['category']=='ai_raw']),hard=stats([r for r in sub if r['category']=='ai_casual_or_humanized']),polished=stats([r for r in sub if r['label']=='ai_edited']),excluding_polish=stats([r for r in sub if r['label']!='ai_edited']))
 short=[r for r in rows if 30<=r['word_count']<=60];priority_genres={g:split([r for r in priority_rows if r['genre']==g]) for g in sorted({r['genre'] for r in priority_rows})}
 models={m:stats([r for r in gen if by[r['id']]['source'].get('model')==m]) for m in sorted({by[r['id']]['source'].get('model') for r in gen})}
 hard={s:stats([r for r in rows if r['category']=='ai_casual_or_humanized' and by[r['id']]['subtype']==s]) for s in sorted({by[r['id']]['subtype'] for r in rows if r['category']=='ai_casual_or_humanized'})}
 errors=[]
 for r in rows:
  if any(flag(r,k)!=(r['label']!='human') for k in ['ours','verify']):
   t=by[r['id']];errors.append(dict(id=r['id'],category=r['category'],genre=r['genre'],subtype=t.get('subtype'),word_count=t['word_count'],priority=priority(t),text=t['text'],source=t['source'],ours=r['ours']['raw']['ai_score'],zero=r['zerogpt']['raw']['data']['fakePercentage'],verify=r['verify']['raw'],detect=r['ours']['raw']))
 ours_only=sum(flag(r,'ours')==(r['label']!='human') and flag(r,'zerogpt')!=(r['label']!='human') for r in nonpol);zero_only=sum(flag(r,'zerogpt')==(r['label']!='human') and flag(r,'ours')!=(r['label']!='human') for r in nonpol);discordant=ours_only+zero_only
 p=min(1,2*sum(math.comb(discordant,i) for i in range(min(ours_only,zero_only)+1))/2**discordant) if discordant else 1
 parity=stats(nonpol)['ours']['correct']>=stats(nonpol)['zerogpt']['correct'] and stats(humans)['ours']['fp']<=stats(humans)['zerogpt']['fp'] and cats['ai_casual_or_humanized']['ours']['flagged']>=cats['ai_casual_or_humanized']['zerogpt']['flagged']
 intervals={name:{k:wilson(s[k]['flagged'],s['n']) for k in KEYS} for name,s in [('human_fp',stats(humans)),('hard_catch',cats['ai_casual_or_humanized']),('short_human_fp',stats([r for r in short if r['label']=='human']))]}
 result=dict(round=n,quality=quality,categories=cats,all=stats(rows),excluding_polish=stats(nonpol),priority=split(priority_rows),priority_genres=priority_genres,short=split(short),bias=bias,models=models,hard_subtypes=hard,parity=parity,ours_only_correct=ours_only,zero_only_correct=zero_only,mcnemar_exact_p=p,wilson95=intervals,errors=errors,health_history=out['health_history'],health_after=out['health_after'],verify_versions=dict(collections.Counter(json.dumps(r['verify']['raw'].get('versions'),sort_keys=True) for r in rows)),devices=dict(collections.Counter(r['ours']['raw'].get('device') for r in rows)),verify_store_hits=dict(collections.Counter(str(r['verify']['raw'].get('from_store')) for r in rows)),detect_verify_score_mismatches=[r['id'] for r in rows if abs(r['ours']['raw']['ai_score']-r['verify']['raw']['ai_score'])>1e-6],started_at=out['started_at'],finished_at=out['finished_at'],texts_file_sha256=out['texts_file_sha256'],local_request_attempts=len(attempts),local_retries=len(attempts)-2*len(rows))
 if n==2:
  prior={r['id']:r for r in old['rows']};result['since_round2']={k:dict(previous=stats(old['rows'])[k],current=stats(rows)[k],fixed=[r['id'] for r in nonpol if flag(prior[r['id']],k)!=(r['label']!='human') and flag(r,k)==(r['label']!='human')],regressed=[r['id'] for r in nonpol if flag(prior[r['id']],k)==(r['label']!='human') and flag(r,k)!=(r['label']!='human')]) for k in ['ours','verify']}
 return result
out={'round3':analyze(3,'results.json'),'round2_rerun':analyze(2,'round2-results.json')}
# Check serial execution across the entire two-dataset run as well.
a=[]
for f in ['results.json','round2-results.json']:
 for r in json.loads((R/f).read_text())['rows']:
  for k in ['ours','verify']:a+=r[k]['attempts']
a.sort(key=lambda x:x['started_at'])
for x,y in zip(a,a[1:]):assert x['finished_at']<=y['started_at']
(R/'phase_b_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for name,s in out.items():
 print(name,'QUALITY',s['quality'],'PARITY',s['parity'],'VERSIONS',s['verify_versions'],'ATTEMPTS',s['local_request_attempts'])
 print('ACCURACY',s['excluding_polish']);print('CATEGORIES',s['categories']);print('PRIORITY',s['priority']);print('SHORT',s['short']);print('BIAS',json.dumps(s['bias']));print('HARD',s['hard_subtypes']);print('MODELS',s['models'])
 print('ERRORS',[(e['id'],e['category'],e['genre'],round(e['ours']*100,2),e['zero'],e['verify']['verdict'],e['text'][:180]) for e in s['errors']])
 if 'since_round2' in s:print('CHANGE',s['since_round2'])
