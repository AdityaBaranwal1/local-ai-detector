import pathlib,json,math,collections,statistics,hashlib
ROOT=pathlib.Path(__file__).resolve().parent
texts={x['id']:x for x in json.loads((ROOT/'texts.json').read_text())}
r=json.loads((ROOT/'results.json').read_text())
assert r.get('finished_at'),'Evaluation is not finished'
rows=r['rows']; assert len(rows)==120
assert all(x['sha256']==texts[x['id']]['sha256'] for x in rows)
CATS=['human_native','human_non_native','ai_raw','ai_casual_or_humanized','ai_polished_human']
DETECTORS=['ours','zerogpt','verify']
def score(row,d):
 if not row[d]['ok']:return None
 if d=='ours':return float(row[d]['raw']['ai_score'])
 if d=='zerogpt':return float(row[d]['raw']['data']['fakePercentage'])/100
 return 1 if row[d]['raw']['verdict']=='FAIL' else 0
def flag(row,d):
 s=score(row,d);return None if s is None else s>=0.5
def truth(row):return row['label']!='human'
def metrics(pool,d):
 valid=[x for x in pool if flag(x,d) is not None]
 tp=sum(truth(x) and flag(x,d) for x in valid);tn=sum(not truth(x) and not flag(x,d) for x in valid)
 fp=sum(not truth(x) and flag(x,d) for x in valid);fn=sum(truth(x) and not flag(x,d) for x in valid)
 return {'n':len(valid),'total':len(pool),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'accuracy':(tp+tn)/len(valid) if valid else None,'specificity':tn/(tn+fp) if tn+fp else None,'sensitivity':tp/(tp+fn) if tp+fn else None,'balanced_accuracy':((tn/(tn+fp)+tp/(tp+fn))/2) if tn+fp and tp+fn else None}
def rate(pool,d):
 vals=[flag(x,d) for x in pool if flag(x,d) is not None];k=sum(vals);n=len(vals)
 if not n:return {'n':0,'flagged':0,'rate':None,'wilson95':None}
 p=k/n;z=1.959963984540054;den=1+z*z/n;center=(p+z*z/(2*n))/den;rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return {'n':n,'flagged':k,'rate':p,'wilson95':[max(0,center-rad),min(1,center+rad)]}
s={'coverage':{d:sum(x[d]['ok'] for x in rows) for d in DETECTORS},'category':{c:{d:rate([x for x in rows if x['category']==c],d) for d in DETECTORS} for c in CATS}}
s['overall']={d:metrics(rows,d) for d in DETECTORS}
base=[x for x in rows if x['category']!='ai_polished_human']
s['excluding_polished']={d:metrics(base,d) for d in DETECTORS}
s['scorecard']={}
for d in DETECTORS:
 h=s['overall'][d]['specificity'];raw=s['category']['ai_raw'][d]['rate'];hard=s['category']['ai_casual_or_humanized'][d]['rate']
 s['scorecard'][d]=10*(.4*h+.2*raw+.4*hard) if None not in (h,raw,hard) else None
pair=[x for x in base if x['ours']['ok'] and x['zerogpt']['ok']]
wins=[x['id'] for x in pair if flag(x,'ours')==truth(x) and flag(x,'zerogpt')!=truth(x)]
losses=[x['id'] for x in pair if flag(x,'ours')!=truth(x) and flag(x,'zerogpt')==truth(x)]
n=len(wins)+len(losses);k=min(len(wins),len(losses));p=min(1,2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1
s['paired']={'n':len(pair),'ours_only_correct':wins,'zerogpt_only_correct':losses,'mcnemar_exact_two_sided_p':p,'accuracy_difference':(len(wins)-len(losses))/len(pair)}
s['at_least_as_good']=bool(s['excluding_polished']['ours']['accuracy']>=s['excluding_polished']['zerogpt']['accuracy'] and s['overall']['ours']['fp']<=s['overall']['zerogpt']['fp'] and s['category']['ai_casual_or_humanized']['ours']['rate']>=s['category']['ai_casual_or_humanized']['zerogpt']['rate'])
s['versions']=dict(collections.Counter(json.dumps(x['verify']['raw'].get('versions'),sort_keys=True) for x in rows if x['verify']['ok']))
s['endpoint_score_disagreements']=[{'id':x['id'],'detect_score':score(x,'ours'),'verify_score':x['verify']['raw'].get('ai_score')} for x in rows if x['ours']['ok'] and x['verify']['ok'] and abs(score(x,'ours')-x['verify']['raw'].get('ai_score',0))>0.0001]
s['by_model']={m:{d:rate([x for x in rows if x['model']==m and x['category']!='ai_polished_human'],d) for d in DETECTORS} for m in sorted({x['model'] for x in rows if x['model']})}
s['hard_subtypes']={st:{d:rate([x for x in rows if texts[x['id']].get('subtype')==st],d) for d in DETECTORS} for st in sorted({t.get('subtype') for t in texts.values() if t['category']=='ai_casual_or_humanized'})}
s['latencies']={d:{'median':statistics.median([x[d]['elapsed_seconds'] for x in rows if x[d]['ok']]),'sum_seconds':sum(x[d]['elapsed_seconds'] for x in rows if x[d]['ok'])} for d in DETECTORS}
s['lengths']={c:{'min':min(x['word_count'] for x in rows if x['category']==c),'median':statistics.median(x['word_count'] for x in rows if x['category']==c),'max':max(x['word_count'] for x in rows if x['category']==c)} for c in CATS}
s['errors']=[{'id':x['id'],'detector':d,'detail':x[d]} for x in rows for d in DETECTORS if not x[d]['ok']]
s['engine_devices']=dict(collections.Counter(x['ours']['raw'].get('device') for x in rows if x['ours']['ok']))
s['ours_errors']=[]
for x in rows:
 if x['ours']['ok'] and flag(x,'ours')!=truth(x):
  t=texts[x['id']]
  s['ours_errors'].append({'id':x['id'],'category':x['category'],'model':x['model'],'subtype':t.get('subtype'),'ours':score(x,'ours'),'zero':score(x,'zerogpt'),'gate':x['verify']['raw'].get('verdict'),'genre':t['genre'],'quote':t['text'][:280],'classes':x['ours']['raw'].get('classes'),'verify_problems':x['verify']['raw'].get('problems')})
(ROOT/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
print('COVERAGE',s['coverage']);print('SCORE',s['scorecard']);print('AT LEAST AS GOOD',s['at_least_as_good'])
for c in CATS:print(c,[(d,s['category'][c][d]['flagged'],s['category'][c][d]['n'],round(100*s['category'][c][d]['rate'],1)) for d in DETECTORS])
print('OVERALL',json.dumps(s['overall'],indent=2));print('EXCLUDING POLISHED',json.dumps(s['excluding_polished'],indent=2));print('PAIRED',s['paired']);print('VERSIONS',s['versions']);print('SCORE DISAGREEMENTS',s['endpoint_score_disagreements']);print('MODELS',json.dumps(s['by_model'],indent=2));print('HARD SUBTYPES',json.dumps(s['hard_subtypes'],indent=2));print('LATENCIES',s['latencies'])
print('FAILURES')
for x in s['ours_errors']:print(json.dumps(x,ensure_ascii=False))
