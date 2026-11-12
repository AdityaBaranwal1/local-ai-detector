import pathlib,json,urllib.request,urllib.error,time,datetime,random,hashlib
R=pathlib.Path(__file__).resolve().parent
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest=json.loads((R/'manifest.json').read_text());assert hashlib.sha256((R/'texts.json').read_bytes()).hexdigest()==manifest['texts_file_sha256']
lastzero=0
ua='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
def call(url,body,zero=False):
 global lastzero
 attempts=[]
 for attempt in range(3):
  if zero:
   delay=5.3-(time.monotonic()-lastzero)
   if delay>0:time.sleep(delay)
  started=time.monotonic();ts=now();headers={'Content-Type':'application/json','User-Agent':ua}
  if zero:headers.update(Origin='https://www.zerogpt.com',Referer='https://www.zerogpt.com/')
  try:
   req=urllib.request.Request(url,data=json.dumps(body).encode(),headers=headers)
   with urllib.request.urlopen(req,timeout=600) as res:status=res.status;raw=json.loads(res.read().decode())
   if zero and (raw.get('success') is not True or 'fakePercentage' not in raw.get('data',{})):raise ValueError('ZeroGPT error '+str(raw)[:800])
   if not zero and url.endswith('/api/detect') and 'ai_score' not in raw:raise ValueError('Missing ai_score '+str(raw)[:800])
   if not zero and url.endswith('/verify') and (raw.get('verdict') not in ('PASS','FAIL') or not raw.get('complete')):raise ValueError('Incomplete verification '+str(raw)[:800])
   attempts.append(dict(started_at=ts,elapsed_seconds=time.monotonic()-started,http_status=status,raw=raw))
   if zero:lastzero=time.monotonic()
   return dict(ok=True,raw=raw,elapsed_seconds=time.monotonic()-started,attempts=attempts)
  except Exception as e:
   err=dict(started_at=ts,elapsed_seconds=time.monotonic()-started,error=repr(e));retry_after=0
   if isinstance(e,urllib.error.HTTPError):
    err.update(http_status=e.code,response_body=e.read().decode(errors='replace')[:4000])
    try:retry_after=float(e.headers.get('Retry-After','0'))
    except ValueError:pass
   attempts.append(err)
   if zero:lastzero=time.monotonic()
   print('REQUEST ERROR',url,attempt+1,repr(e),flush=True)
   if attempt<2:time.sleep(max(retry_after,15*(attempt+1)))
 return dict(ok=False,attempts=attempts)
def run(source,output,dataset):
 texts=json.loads(source.read_text());path=R/output
 if path.exists():results=json.loads(path.read_text())
 else:
  results=dict(dataset=dataset,started_at=now(),texts_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),manifest=manifest if dataset=='fresh' else json.loads((R.parent/'round-1/manifest.json').read_text()),thresholds=dict(ours=0.5,zerogpt_percent=50,verify='FAIL is blocked'),rows=[])
 assert hashlib.sha256(source.read_bytes()).hexdigest()==results['texts_file_sha256']
 rows={x['id']:x for x in results['rows']}
 def save():
  tmp=path.with_suffix('.json.partial');tmp.write_text(json.dumps(results,ensure_ascii=False,indent=2));tmp.replace(path)
 try:results['health_before']=json.load(urllib.request.urlopen('http://127.0.0.1:8323/healthz',timeout=30))
 except Exception as e:results['health_before_error']=repr(e)
 order=list(texts);random.Random(2026111202).shuffle(order);results['order']=[x['id'] for x in order];save()
 for i,t in enumerate(order):
  sha=hashlib.sha256(t['text'].encode()).hexdigest();assert sha==t['sha256']
  row=rows.get(t['id'])
  if row is None:
   row=dict(id=t['id'],sha256=sha,label=t['label'],category=t['category'],genre=t['genre'],word_count=t['word_count'],model=t['source'].get('model'));rows[t['id']]=row;results['rows'].append(row)
  for key,url,body in [('ours','http://127.0.0.1:8323/api/detect',{'text':t['text']}),('verify','http://127.0.0.1:8323/verify',{'text':t['text']}),('zerogpt','https://api.zerogpt.com/api/detect/detectText',{'input_text':t['text']})]:
   if key in row:continue
   row[key]=call(url,body,key=='zerogpt');save()
  print(dataset,f'{i+1:03}/{len(texts)}',t['id'],t['category'],'ours='+(str(row['ours']['raw']['ai_score']) if row['ours']['ok'] else 'ERROR'),'verify='+(row['verify']['raw']['verdict'] if row['verify']['ok'] else 'ERROR'),'ZeroGPT='+(str(row['zerogpt']['raw']['data']['fakePercentage']) if row['zerogpt']['ok'] else 'ERROR'),flush=True)
 results['finished_at']=now()
 try:results['health_after']=json.load(urllib.request.urlopen('http://127.0.0.1:8323/healthz',timeout=30))
 except Exception as e:results['health_after_error']=repr(e)
 save();print(dataset,'EVALUATION COMPLETE',results['finished_at'],flush=True)
run(R/'texts.json','results.json','fresh')
run(R.parent/'round-1/texts.json','round1-results.json','round1_rerun')
print('ALL EVALUATIONS COMPLETE',now(),flush=True)
