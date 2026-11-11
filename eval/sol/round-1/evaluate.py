import pathlib,json,urllib.request,urllib.error,time,datetime,random,hashlib,traceback
ROOT=pathlib.Path(__file__).resolve().parent
texts=json.loads((ROOT/'texts.json').read_text())
manifest=json.loads((ROOT/'manifest.json').read_text())
assert hashlib.sha256((ROOT/'texts.json').read_bytes()).hexdigest()==manifest['texts_file_sha256']
path=ROOT/'results.json'
if path.exists():results=json.loads(path.read_text())
else:results={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest':manifest,'thresholds':{'ours':0.5,'zerogpt_percent':50,'verify':'FAIL is flagged; default max_ai=0.30 plus strong tells'},'rows':[]}
rows={x['id']:x for x in results['rows']}
def save():
 tmp=path.with_suffix('.json.partial');tmp.write_text(json.dumps(results,ensure_ascii=False,indent=2));tmp.replace(path)
lastzero=0
ua='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
def call(url,body,zero=False):
 global lastzero
 attempts=[]
 for attempt in range(3):
  if zero:
   delay=5.2-(time.monotonic()-lastzero)
   if delay>0:time.sleep(delay)
  started=time.monotonic();now=datetime.datetime.now(datetime.timezone.utc).isoformat()
  headers={'Content-Type':'application/json','User-Agent':ua}
  if zero:headers.update(Origin='https://www.zerogpt.com',Referer='https://www.zerogpt.com/')
  try:
   req=urllib.request.Request(url,data=json.dumps(body).encode(),headers=headers)
   with urllib.request.urlopen(req,timeout=300) as response:
    status=response.status;raw=response.read().decode();rh=dict(response.headers)
   parsed=json.loads(raw)
   if zero and (parsed.get('success') is not True or 'fakePercentage' not in parsed.get('data',{})):raise ValueError('API error response: '+raw[:500])
   if not zero and url.endswith('/api/detect') and 'ai_score' not in parsed:raise ValueError('No ai_score')
   if not zero and url.endswith('/verify') and (parsed.get('verdict') not in ('PASS','FAIL') or not parsed.get('complete')):raise ValueError('Incomplete verification: '+raw[:500])
   attempts.append({'started_at':now,'elapsed_seconds':time.monotonic()-started,'http_status':status,'raw':parsed})
   if zero:lastzero=time.monotonic()
   return {'ok':True,'raw':parsed,'elapsed_seconds':time.monotonic()-started,'attempts':attempts}
  except Exception as e:
   error={'started_at':now,'elapsed_seconds':time.monotonic()-started,'error':repr(e)}
   retry_after=0
   if isinstance(e,urllib.error.HTTPError):
    error['http_status']=e.code;error['response_body']=e.read().decode(errors='replace')[:4000]
    try:retry_after=float(e.headers.get('Retry-After','0'))
    except ValueError:pass
   attempts.append(error)
   if zero:lastzero=time.monotonic()
   print('REQUEST ERROR',url,attempt+1,repr(e),flush=True)
   if attempt<2:time.sleep(max(retry_after,10*(attempt+1)))
 return {'ok':False,'attempts':attempts}
order=list(texts);random.Random(683109).shuffle(order)
results['order']=[x['id'] for x in order]
save()
for index,t in enumerate(order):
 row=rows.get(t['id'])
 if row is None:
  row={'id':t['id'],'sha256':t['sha256'],'label':t['label'],'category':t['category'],'genre':t['genre'],'word_count':t['word_count'],'model':t['source'].get('model')};rows[t['id']]=row;results['rows'].append(row)
 for key,url,body in [('ours','http://127.0.0.1:8323/api/detect',{'text':t['text']}),('verify','http://127.0.0.1:8323/verify',{'text':t['text']}),('zerogpt','https://api.zerogpt.com/api/detect/detectText',{'input_text':t['text']})]:
  if key in row:continue
  row[key]=call(url,body,key=='zerogpt');save()
 print(f"{index+1:03}/120 {t['id']} {t['category']} ours="+(str(row['ours']['raw']['ai_score']) if row['ours']['ok'] else 'ERROR')+' verify='+(row['verify']['raw']['verdict'] if row['verify']['ok'] else 'ERROR')+' ZeroGPT='+(str(row['zerogpt']['raw']['data']['fakePercentage']) if row['zerogpt']['ok'] else 'ERROR'),flush=True)
results['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
try:results['health_after']=json.load(urllib.request.urlopen('http://127.0.0.1:8323/healthz',timeout=30))
except Exception as e:results['health_after_error']=repr(e)
save();print('EVALUATION COMPLETE',results['finished_at'],flush=True)
