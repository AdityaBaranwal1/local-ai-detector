import pathlib,json,urllib.request,urllib.error,time,re,hashlib,datetime,collections
import yaml
R=pathlib.Path(__file__).resolve().parent
c=yaml.safe_load((pathlib.Path.home()/'docker/cliproxyapi/config.yaml').read_text());KEY=c.get('api-keys',c.get('access',{}).get('api-keys'))[0]
MODELS=['claude-haiku-5-5','gemini-3-flash','gpt-6-luna'];humans=json.loads((R/'humans.json').read_text());OUT=R/'generation';OUT.mkdir(exist_ok=True)
def generate(model,tag,specs):
 path=OUT/(model+'-'+tag+'.json')
 if path.exists():return json.loads(path.read_text())['parsed']
 prompt='Create synthetic writing for an authorized private AI-detector evaluation. These will never be sent as real messages or applications. Invent all facts except supplied public pre-2022 editing sources. Return ONLY a JSON array of objects with id and text, exactly one per spec. No markdown fences or explanatory commentary. Each sample 100–220 words unless a spec asks for 280–420. No placeholders. Make samples genuinely varied. Follow style instructions closely.\n'+json.dumps(specs,ensure_ascii=False)
 body={'model':model,'messages':[{'role':'user','content':prompt}],'max_tokens':9000,'temperature':0.9};errors=[]
 for attempt in range(3):
  start=time.monotonic()
  try:
   req=urllib.request.Request('http://127.0.0.1:8317/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
   raw=json.load(urllib.request.urlopen(req,timeout=600));content=raw['choices'][0]['message']['content'];parsed=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',content.strip()));assert isinstance(parsed,list) and len(parsed)==len(specs)
   byid={x['id']:x for x in parsed};parsed=[byid[x['id']] for x in specs]
   for x in parsed:assert 60<=len(x['text'].split())<=500,(x['id'],len(x['text'].split()))
   path.write_text(json.dumps(dict(model=model,tag=tag,request=body,response=raw,parsed=parsed,duration_seconds=time.monotonic()-start,generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),ensure_ascii=False,indent=2));print('GENERATED',model,tag,[(x['id'],len(x['text'].split())) for x in parsed],flush=True);return parsed
  except Exception as e:
   errors.append(repr(e));print('RETRY',model,tag,repr(e),flush=True);time.sleep(10*(attempt+1))
 (OUT/(model+'-'+tag+'-error.json')).write_text(json.dumps(errors));raise RuntimeError((model,tag,errors))
ai=[]
def record(model,tag,specs,parsed,cat):
 for spec,p in zip(specs,parsed):
  x=dict(id=p['id'],label='ai_edited' if cat=='ai_polished_human' else 'ai',category=cat,genre=spec['genre'],subtype=spec.get('subtype','raw'),text=p['text'],word_count=len(p['text'].split()),source=dict(model=model,generation_file=f'generation/{model}-{tag}.json',prompt_spec=spec))
  if 'parent_id' in spec:x['parent_id']=spec['parent_id']
  ai.append(x)
rawtopics=[('cold_outreach','A sound technician offers a local theatre help with patchy speech audibility. Mention visiting a public performance and ask whether a short test would help.'),('followup_email','Follow up after a facilities assistant interview. Mention the conversation about lost access cards and your experience keeping a simple check-out log.'),('application_answer','Answer why you want a billing support position at a fictional community internet provider. Concrete relevant work at a repair counter, no inflated claims.'),('cover_letter','Apply for a junior laboratory stockroom coordinator role at a fictional college. Describe inventory work and following safety procedures. 280–420 words.'),('linkedin_post','Describe learning to document handovers after a summer placement at a small packaging company. Include an actual mundane example and one uncertainty.'),('scientific_encyclopedic','Explain how an urban wetland handles heavy rain and what limits its effectiveness. Precise, neutral encyclopedic prose, no headings, 280–420 words.')]
styles=[('lowercase','followup_email','all lowercase, follow up on a promised rota change at a cafe. Slightly rambling, no stock closing, conversational and concrete.'),('non_native_imitation','application_answer','Answer why you fit a junior procurement role, imitating an intermediate English learner. Believable tense/article errors without caricaturing a nationality. Include a supplier delay example.'),('casual','review','A casual mixed review of a cheap mechanical kitchen timer; a specific good feature and an annoying fault, no tidy conclusion.'),('lowercase','linkedin_message','all lowercase request to an old classmate for advice about a trainee surveyor vacancy; hesitant, uneven rhythm, one brief aside.'),('non_native_imitation','work_email','Imitate an intermediate English learner writing to a manager about mixing up two room bookings. Natural small grammar mistakes, practical remedy, no stock thanks.'),('casual','short_story','A restrained first-person story about finding an old bus ticket in a borrowed coat. Uneven rhythm and concrete detail, no grand revelation.'),('humanize_paraphrase','cold_outreach','Rewrite the supplied AI draft into understated everyday professional prose. Preserve facts but avoid generic openings, symmetrical sentences and neat conclusions.'),('humanize_paraphrase','cover_letter','Humanize the supplied AI cover letter: ordinary direct language, uneven rhythm, realistic self-doubt, keep factual claims and core application intent.'),('humanize_paraphrase','scientific_encyclopedic','Paraphrase supplied AI science prose into conversational explanation, with fragments and an aside. Keep factual claims, no added personal history.')]
polish=[['R2H001','R2H011','R2H026','R2H044'],['R2H004','R2H020','R2H058'],['R2H008','R2H035','R2H052']]
for mi,model in enumerate(MODELS):
 pref=['R2K','R2G','R2L'][mi];specs=[dict(id=f'{pref}R{i+1:02}',genre=g,task=task) for i,(g,task) in enumerate(rawtopics)];raw=generate(model,'raw',specs);record(model,'raw',specs,raw,'ai_raw');count=8 if mi==0 else 9
 for start in [0,6]:
  specs=[]
  for i in range(start,min(start+6,count)):
   st,g,task=styles[i];sp=dict(id=f'{pref}H{i+1:02}',genre=g,subtype=st,task=task)
   if st=='humanize_paraphrase':
    parent={6:0,7:3,8:5}[i];sp.update(parent_id=raw[parent]['id'],source_text=raw[parent]['text'])
   specs.append(sp)
  tag=f'hard-{start}';got=generate(model,tag,specs);record(model,tag,specs,got,'ai_casual_or_humanized')
 specs=[]
 for i,hid in enumerate(polish[mi]):
  h=next(x for x in humans if x['id']==hid);specs.append(dict(id=f'{pref}P{i+1:02}',genre=h['genre'],subtype='polished_human',parent_id=hid,source_text=h['text'],task='Polish the supplied public pre-2022 human text for clarity and fluency. Preserve meaning and all claims, correct errors, do not add details.'))
 got=generate(model,'polished',specs);record(model,'polished',specs,got,'ai_polished_human')
for i,x in enumerate(json.loads((R/'self_ai.json').read_text())):
 x.update(id=f'R2S{i+1:03}',label='ai',source=dict(model='gpt-6.1-sol',method='Written directly by independent reviewer before detector testing',generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),word_count=len(x['text'].split()));ai.append(x)
texts=humans+ai;assert len(texts)==120
oldhash={hashlib.sha256(x['text'].encode()).hexdigest() for x in json.loads((R.parent/'round-1/texts.json').read_text())}
frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
for x in texts:
 x['frozen_at']=frozen_at
 x['sha256']=hashlib.sha256(x['text'].encode()).hexdigest();assert x['sha256'] not in oldhash;assert 60<=x['word_count']<=500
assert len({x['sha256'] for x in texts})==120
(R/'texts.json').write_text(json.dumps(texts,ensure_ascii=False,indent=2));manifest=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),texts_file_sha256=hashlib.sha256((R/'texts.json').read_bytes()).hexdigest(),counts=dict(collections.Counter(x['category'] for x in texts)),note='All selections and generations precede any detector call; no texts reused from round 1; no outcome-dependent replacement.');(R/'manifest.json').write_text(json.dumps(manifest,indent=2));print('FROZEN',manifest,flush=True)
