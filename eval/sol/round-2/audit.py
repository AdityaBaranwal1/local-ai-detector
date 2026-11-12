import pathlib,json,datetime,collections,hashlib,math
R=pathlib.Path(__file__).resolve().parent
G={'cold_outreach','followup_email','work_email','reference_request','linkedin_message','linkedin_post','application_answer','cover_letter'}
out={'priority_slice_note':'Post-hoc diagnostic motivated by user clarification during the frozen run; not part of the preregistered quality score. Human sources are job/work exam proxies, not authentic LinkedIn records.','datasets':{}}
zero=[];retries=collections.Counter();versions=collections.Counter()
for ds,tf,rf in [('fresh',R/'texts.json',R/'results.json'),('round1_rerun',R.parent/'round-1/texts.json',R/'round1-results.json')]:
 texts=json.loads(tf.read_text());ts={t['id']:t for t in texts};results=json.loads(rf.read_text());rs=results['rows'];assert len(rs)==120
 for r in rs:
  assert r['sha256']==hashlib.sha256(ts[r['id']]['text'].encode()).hexdigest()
  for k in ['ours','verify','zerogpt']:
   assert r[k]['ok'];retries[k]+=len(r[k]['attempts'])-1
   if k=='zerogpt':
    for a in r[k]['attempts']:zero.append((datetime.datetime.fromisoformat(a['started_at']).timestamp(),a['elapsed_seconds']))
  versions[json.dumps(r['verify']['raw']['versions'],sort_keys=True)]+=1
 def flag(r,k):return r[k]['raw']['ai_score']>=.5 if k=='ours' else r[k]['raw']['data']['fakePercentage']>=50 if k=='zerogpt' else r[k]['raw']['verdict']=='FAIL'
 slices={}
 for title,pred in [('job_work_human_proxies',lambda t:t['label']=='human' and t['genre'] in G),('professional_ai',lambda t:t['label']=='ai' and t['genre'] in G),('professional_hard_ai',lambda t:t['category']=='ai_casual_or_humanized' and t['genre'] in G),('professional_natural_casual_ai',lambda t:t['category']=='ai_casual_or_humanized' and t['genre'] in G and t.get('subtype')=='casual'),('linkedin_ai',lambda t:t['label']=='ai' and t['genre'] in ['linkedin_message','linkedin_post'])]:
  rows=[r for r in rs if pred(ts[r['id']])];slices[title]={'n':len(rows),'flagged':{k:sum(flag(r,k) for r in rows) for k in ['ours','zerogpt','verify']},'ids':[r['id'] for r in rows]}
 # Independently recompute weighted score, directly from raw data, without analyze.py.
 score={}
 for k in ['ours','zerogpt']:
  h=[r for r in rs if r['label']=='human'];raw=[r for r in rs if r['category']=='ai_raw'];hard=[r for r in rs if r['category']=='ai_casual_or_humanized'];score[k]=10*(.4*sum(not flag(r,k) for r in h)/len(h)+.2*sum(flag(r,k) for r in raw)/len(raw)+.4*sum(flag(r,k) for r in hard)/len(hard))
 out['datasets'][ds]={'priority_slices':slices,'independent_quality':score,'endpoint_successes':{k:sum(r[k]['ok'] for r in rs) for k in ['ours','verify','zerogpt']}}
zero.sort();out['zerogpt_requests_including_retries']=len(zero);out['minimum_zerogpt_pause_after_previous_completion_seconds']=min(zero[i][0]-(zero[i-1][0]+zero[i-1][1]) for i in range(1,len(zero)));out['retry_counts']=dict(retries);out['verify_versions']=dict(versions)
a={r['id']:r for r in json.loads((R.parent/'round-1/results.json').read_text())['rows']};b={r['id']:r for r in json.loads((R/'round1-results.json').read_text())['rows']};changes={}
for k in ['ours','verify','zerogpt']:
 def flag(r):return r[k]['raw']['ai_score']>=.5 if k=='ours' else r[k]['raw']['verdict']=='FAIL' if k=='verify' else r[k]['raw']['data']['fakePercentage']>=50
 fixed=[];regressed=[]
 for id in a:
  old=flag(a[id])==(a[id]['label']!='human');new=flag(b[id])==(b[id]['label']!='human')
  if old!=new:(fixed if new else regressed).append(id)
 changes[k]={'fixed':fixed,'regressed':regressed}
out['round1_paired_changes']=changes
fresh=json.loads((R/'texts.json').read_text());old=json.loads((R.parent/'round-1/texts.json').read_text());assert not ({hashlib.sha256(t['text'].encode()).hexdigest() for t in fresh}&{hashlib.sha256(t['text'].encode()).hexdigest() for t in old});out['fresh_vs_round1_exact_overlap']=0
m=json.loads((R/'manifest.json').read_text());assert hashlib.sha256((R/'texts.json').read_bytes()).hexdigest()==m['texts_file_sha256'];out['frozen_manifest_verified']=True
out['wilson_95']={}
for name,k,n in [('fresh_human_fpr',1,60),('fresh_nonnative_fpr',0,30),('fresh_hard_sensitivity',26,30),('fresh_casual_sensitivity',5,8),('fresh_zerogpt_hard_sensitivity',29,30)]:
 z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;out['wilson_95'][name]=[mid-rad,mid+rad]
(R/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
