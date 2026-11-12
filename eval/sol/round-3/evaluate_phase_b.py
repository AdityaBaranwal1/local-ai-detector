import pathlib,json,hashlib,urllib.request,urllib.error,time,datetime,random,math
R=pathlib.Path(__file__).resolve().parent
EXPECTED={3:'20efbed370ffba61f7434286652e8ba5ffdf4e72f52af12767235e349c27abfd',2:'45f4711b900d4f2ddf4ee473aa5459edf10642be7aa4469b4e595d46e00ab247'}
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,j):
 tmp=p.with_suffix('.json.partial');tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def health():
 with urllib.request.urlopen('http://127.0.0.1:8323/healthz',timeout=30) as f:
  body=f.read().decode();return dict(at=now(),http_status=f.status,raw_http_body=body,raw=json.loads(body))
def call(endpoint,text):
 attempts=[]
 for i in range(3):
  a=dict(started_at=now());tic=time.monotonic()
  try:
   q=urllib.request.Request('http://127.0.0.1:8323'+endpoint,data=json.dumps({'text':text}).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(q,timeout=600) as f:a.update(http_status=f.status,raw_http_body=f.read().decode())
   a.update(finished_at=now(),elapsed_seconds=time.monotonic()-tic);raw=json.loads(a['raw_http_body'])
   if endpoint=='/api/detect':assert math.isfinite(raw['ai_score']) and 0<=raw['ai_score']<=1
   else:assert raw.get('complete') and raw.get('verdict') in ('PASS','FAIL'),raw
   attempts.append(a);return dict(ok=True,raw=raw,attempts=attempts)
  except Exception as e:
   a.update(finished_at=now(),elapsed_seconds=time.monotonic()-tic,error=repr(e))
   if isinstance(e,urllib.error.HTTPError):a.update(http_status=e.code,raw_http_body=e.read().decode(errors='replace'))
   attempts.append(a);print('ERROR',endpoint,repr(e),flush=True)
   if i<2:time.sleep(15*(i+1))
 return dict(ok=False,attempts=attempts)
# Validate both immutable datasets before the first model request.
for n,h in EXPECTED.items():
 p=R.parent/f'round-{n}'/'texts.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 for t in json.loads(p.read_text()):assert hashlib.sha256(t['text'].encode()).hexdigest()==t['sha256']
for n in [3,2]:
 folder=R.parent/f'round-{n}';texts=json.loads((folder/'texts.json').read_text());output=R/('results.json' if n==3 else 'round2-results.json')
 if output.exists():out=json.loads(output.read_text());assert out['texts_file_sha256']==EXPECTED[n]
 else:out=dict(round=n,phase='B',started_at=now(),texts_file_sha256=EXPECTED[n],thresholds={'ours':.5,'zerogpt_percent':50},rows=[],health_history=[])
 before=health();out['health_history'].append(before);engine=before['raw']['engine'];print('ROUND',n,'HEALTH',json.dumps(before),flush=True)
 assert '+stack@0d0c5cda' in engine,'Unexpected deployed engine; stop rather than evaluate an unrequested build'
 old=json.loads((folder/('zerogpt.json' if n==3 else 'results.json')).read_text());assert old['texts_file_sha256']==EXPECTED[n]
 oldby={r['id']:r for r in old['rows']};by={r['id']:r for r in out['rows']}
 order=list(texts);random.Random(2026111203+n).shuffle(order);out['order']=[t['id'] for t in order];save(output,out)
 for i,t in enumerate(order):
  row=by.get(t['id'])
  if row is None:
   z=oldby[t['id']];assert z['sha256']==t['sha256']
   zr=z if n==3 else z['zerogpt'];assert zr['ok']
   row=dict(id=t['id'],sha256=t['sha256'],label=t['label'],category=t['category'],genre=t['genre'],word_count=t['word_count'],model=t['source'].get('model'),zerogpt=zr,zerogpt_reused_from=str(folder/('zerogpt.json' if n==3 else 'results.json')))
   out['rows'].append(row);by[t['id']]=row
  for key,ep in [('ours','/api/detect'),('verify','/verify')]:
   if not row.get(key,{}).get('ok'):
    row[key]=call(ep,t['text']);save(output,out)
   assert row[key]['ok'],(t['id'],key,'Persistent endpoint error; output saved')
  v=row['verify']['raw'].get('versions',{});assert v.get('engine')==engine,(t['id'],v,engine)
  print('ROUND',n,f'{i+1}/{len(texts)}',t['id'],'ours',row['ours']['raw']['ai_score'],'verify',row['verify']['raw']['verdict'],flush=True)
 out['health_after']=health();assert out['health_after']['raw']['engine']==engine
 assert hashlib.sha256((folder/'texts.json').read_bytes()).hexdigest()==EXPECTED[n]
 out.update(finished_at=now(),complete=len(out['rows'])==len(texts) and all(r['ours']['ok'] and r['verify']['ok'] for r in out['rows']));save(output,out)
 print('COMPLETE ROUND',n,out['finished_at'],flush=True)
