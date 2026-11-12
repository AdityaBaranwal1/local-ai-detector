import pathlib,json,hashlib,urllib.request,urllib.error,time,datetime,random,email.utils,math
R=pathlib.Path(__file__).resolve().parent
URL='https://api.zerogpt.com/api/detect/detectText'
assert URL.startswith('https://api.zerogpt.com/')
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
T=json.loads((R/'texts.json').read_text());M=json.loads((R/'manifest.json').read_text());HASH=hashlib.sha256((R/'texts.json').read_bytes()).hexdigest();assert HASH==M['texts_file_sha256']
P=R/'zerogpt.json'
if P.exists():Z=json.loads(P.read_text());assert Z['texts_file_sha256']==HASH
else:Z=dict(round=3,phase='A',endpoint=URL,started_at=now(),frozen_at=M['frozen_at'],texts_file_sha256=HASH,threshold_percent=50,min_seconds_after_prior_completion=5.3,local_service_called=False,rows=[])
def save():
 tmp=P.with_suffix('.json.partial');tmp.write_text(json.dumps(Z,ensure_ascii=False,indent=2)+'\n');tmp.replace(P)
rows={x['id']:x for x in Z['rows']};last_end=0.0
for row in Z['rows']:
 for a in row.get('attempts',[]):
  if a.get('finished_at'):last_end=max(last_end,datetime.datetime.fromisoformat(a['finished_at']).timestamp())
order=list(T);random.Random(2026111203).shuffle(order);Z['order']=[x['id'] for x in order];save()
headers={'Content-Type':'application/json','Origin':'https://www.zerogpt.com','Referer':'https://www.zerogpt.com/','User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'}
for i,t in enumerate(order):
 assert hashlib.sha256(t['text'].encode()).hexdigest()==t['sha256']
 row=rows.get(t['id'])
 if row is None:
  row=dict(id=t['id'],sha256=t['sha256'],category=t['category'],label=t['label'],genre=t['genre'],priority_genre=t['priority_genre'],short_30_60=t['short_30_60'],word_count=t['word_count'],model=t['source'].get('model'),ok=False,attempts=[]);Z['rows'].append(row);rows[t['id']]=row
 if row['ok']:continue
 stop=False
 for attempt in range(4):
  delay=max(0,5.3-(time.time()-last_end))
  if delay:time.sleep(delay)
  tic=time.monotonic();a=dict(started_at=now(),http_status=None);retry_after=0
  try:
   req=urllib.request.Request(URL,data=json.dumps({'input_text':t['text']}).encode(),headers=headers)
   with urllib.request.urlopen(req,timeout=120) as res:
    a['http_status']=res.status;a['response_headers']=dict(res.headers);a['raw_http_body']=res.read().decode(errors='replace')
   raw=json.loads(a['raw_http_body']);a['raw']=raw
   if raw.get('success') is not True:raise ValueError('API success was not true')
   score=float(raw['data']['fakePercentage']);assert math.isfinite(score) and 0<=score<=100
   row.update(ok=True,score_percent=score,flagged_ai=score>=50,raw=raw)
  except Exception as e:
   a['error']=repr(e)
   if isinstance(e,urllib.error.HTTPError):
    a['http_status']=e.code;a['response_headers']=dict(e.headers);a['raw_http_body']=e.read().decode(errors='replace')
    try:retry_after=float(e.headers.get('Retry-After','0'))
    except ValueError:
     try:retry_after=max(0,email.utils.parsedate_to_datetime(e.headers['Retry-After']).timestamp()-time.time())
     except Exception:pass
    if e.code in (401,403):stop=True
   print('ERROR',t['id'],'attempt',attempt+1,a['error'],'status',a['http_status'],flush=True)
  finally:
   a['finished_at']=now();a['elapsed_seconds']=time.monotonic()-tic;last_end=time.time();row['attempts'].append(a);save()
  if stop:
   Z['stopped_at']=now();Z['stop_reason']='Access denial/challenge; no bypass attempted';save();raise SystemExit(3)
  if row['ok']:break
  if attempt<3:time.sleep(max(20*(2**attempt),retry_after))
 print(f'{i+1:03}/{len(T)}',t['id'],t['category'],'ZeroGPT='+str(row.get('score_percent','ERROR')),'ok='+str(row['ok']),flush=True)
assert hashlib.sha256((R/'texts.json').read_bytes()).hexdigest()==HASH
Z['finished_at']=now();Z['completed_count']=sum(r['ok'] for r in Z['rows']);Z['complete']=Z['completed_count']==len(T);save()
print('ZEROGPT PHASE A COMPLETE',Z['completed_count'],'/',len(T),Z['finished_at'],flush=True)
if not Z['complete']:raise SystemExit(2)
