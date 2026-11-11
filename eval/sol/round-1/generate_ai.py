import pathlib,json,urllib.request,urllib.error,time,re,hashlib,datetime,collections
import yaml
ROOT=pathlib.Path(__file__).resolve().parent
KEY=yaml.safe_load(pathlib.Path('/home/you/docker/cliproxyapi/config.yaml').read_text())['access']['api-keys'][0]
humans=json.loads((ROOT/'humans.json').read_text())
MODELS=['claude-haiku-5-5','gemini-3-flash','gpt-oss-120b-medium']
OUT=ROOT/'generation';OUT.mkdir(exist_ok=True)
def generate(model,tag,specs):
 path=OUT/(model+'-'+tag+'.json')
 if path.exists():return json.loads(path.read_text())['parsed']
 prompt='Generate synthetic writing samples for a private, authorized AI-detector evaluation. None is to be sent as an actual application or message. Use invented facts only unless editing supplied public historical prose. Return ONLY a JSON array, one object per requested sample, each with keys id and text. No markdown fences. Each text must have 90–220 words, be self-contained, and contain no bracket placeholders. Follow every style instruction. Do not explain what you did.\n\n'+json.dumps(specs,ensure_ascii=False)
 body={'model':model,'messages':[{'role':'user','content':prompt}],'max_tokens':7000,'temperature':0.9}
 errors=[]
 for attempt in range(3):
  started=time.monotonic()
  try:
   req=urllib.request.Request('http://127.0.0.1:8317/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
   raw=json.load(urllib.request.urlopen(req,timeout=360))
   content=raw['choices'][0]['message']['content']
   s=re.sub(r'^```(?:json)?\s*|\s*```$','',content.strip())
   parsed=json.loads(s)
   assert isinstance(parsed,list) and len(parsed)==len(specs),(len(parsed),len(specs))
   byid={x['id']:x for x in parsed}
   parsed=[byid[x['id']] for x in specs]
   for x in parsed:assert 60<=len(x['text'].split())<=500,(x['id'],len(x['text'].split()))
   path.write_text(json.dumps({'model':model,'tag':tag,'request':body,'response':raw,'parsed':parsed,'duration_seconds':time.monotonic()-started},ensure_ascii=False,indent=2))
   print('Generated',model,tag,[(x['id'],len(x['text'].split())) for x in parsed],flush=True)
   return parsed
  except Exception as e:
   errors.append(repr(e));print('Generation retry',model,tag,repr(e),flush=True);time.sleep(8*(attempt+1))
 (OUT/(model+'-'+tag+'-error.json')).write_text(json.dumps(errors));raise RuntimeError((model,tag,errors))
ai=[]
def record(model,specs,parsed,category):
 for spec,p in zip(specs,parsed):
  ai.append(dict(id=spec['id'],label='ai_edited' if category=='ai_polished_human' else 'ai',category=category,genre=spec['genre'],subtype=spec.get('subtype','raw'),text=p['text'],word_count=len(p['text'].split()),source={'model':model,'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'generation_file':model+'-'+spec['batch']+'.json','prompt_spec':spec},**({'parent_id':spec['parent_id']} if 'parent_id' in spec else {})))
rawtopics=[
 ('cold_outreach','A freelance technical writer offers to improve a small software company onboarding guide. Mention one concrete observation and ask for a short call.'),
 ('followup_email','Follow up after an interview for a logistics coordinator role. Refer to the discussion of late deliveries and say what experience would help.'),
 ('application_answer','Answer: Tell us about a time you handled conflicting priorities. Invent an ordinary junior analyst example, include a measurable result.'),
 ('cover_letter','Apply for a junior data analyst role at a fictional bicycle retailer. Explain a spreadsheet project and relevant part-time work without inflated claims.'),
 ('linkedin_post','A short professional post about learning to ask better questions during a first internship, with one specific example.'),
 ('essay','Explain whether local libraries should offer tool lending alongside books; consider a practical objection.')]
styles=[
 ('casual','work_email','Tell a teammate a dashboard refresh will be late because the source CSV changed. Sound naturally casual, a little uneven, use contractions, no corporate buzzwords.'),
 ('lowercase','application_answer','All lowercase, informal but credible answer to why a small customer support company interests you. Mention liking the actual product and helping at a shop. Avoid tidy essay structure.'),
 ('non_native_imitation','cover_letter','Imitate an intermediate non-native English writer applying for a hotel receptionist job. Include a few believable article/preposition errors, maintain coherent content, do not caricature a nationality.'),
 ('casual','forum_post','A very informal forum post about a confusing job interview task asking for a whole weekend of unpaid work. Give concrete details and ask one question.'),
 ('lowercase','linkedin_message','All lowercase message asking a former coworker for an introduction to a hiring manager. Conversational, slight digressions, no placeholders, no signature.'),
 ('non_native_imitation','work_email','An intermediate English learner explains a small scheduling mistake to a supervisor and offers a practical fix. Keep a few natural grammar mistakes.'),
 ('humanize_paraphrase','followup_email','Paraphrase the supplied AI text into conversational, less polished prose. Keep the facts, vary sentence rhythm, remove stock transitions. Do not mention the rewrite.'),
 ('humanize_paraphrase','cover_letter','Paraphrase the supplied AI text in a less uniform, everyday voice, preserving the facts. No corporate filler, no neat rhetorical conclusion.'),
 ('humanize_paraphrase','essay','Rewrite the supplied AI text into everyday prose with uneven rhythm and one brief aside, preserving its factual claims. Do not mention AI.')]
polish_ids=[['H001','H014','H043','H051'],['H021','H048','H057'],['H013','H045','H060']]
for mi,model in enumerate(MODELS):
 prefix=['K','G','O'][mi]
 specs=[dict(id=f'{prefix}R{i+1:02}',genre=g,task=task,batch='raw') for i,(g,task) in enumerate(rawtopics)]
 parsed=generate(model,'raw',specs);record(model,specs,parsed,'ai_raw')
 count=8 if mi==0 else 9
 for start in [0,6]:
  specs=[]
  for i in range(start,min(start+6,count)):
   st,g,task=styles[i];spec=dict(id=f'{prefix}H{i+1:02}',genre=g,subtype=st,task=task,batch=f'hard-{start}')
   if st=='humanize_paraphrase':
    parent=[1,3,5][i-6];spec['source_text']=parsed[parent]['text'];spec['parent_id']=parsed[parent]['id']
   specs.append(spec)
  got=generate(model,f'hard-{start}',specs);record(model,specs,got,'ai_casual_or_humanized')
 specs=[]
 for i,hid in enumerate(polish_ids[mi]):
  h=next(x for x in humans if x['id']==hid)
  specs.append(dict(id=f'{prefix}P{i+1:02}',genre=h['genre'],subtype='polished_human',parent_id=hid,batch='polished',task='Polish this public pre-2022 human text into fluent modern professional English. Preserve all facts and intent. Correct mistakes, improve clarity and flow, but do not add claims. Return only the rewritten text in its JSON object.',source_text=h['text']))
 got=generate(model,'polished',specs);record(model,specs,got,'ai_polished_human')
selfs=json.loads((ROOT/'self_ai.json').read_text())
for i,d in enumerate(selfs):
 d.update(id=f'S{i+1:03}',label='ai',source={'model':'gpt-6.1-sol','method':'written directly by independent reviewer, no detector feedback','generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},word_count=len(d['text'].split()))
 ai.append(d)
texts=humans+ai
assert len(texts)==120,len(texts)
assert len({hashlib.sha256(t['text'].encode()).hexdigest() for t in texts})==120
assert all(60<=t['word_count']<=500 for t in texts)
for t in texts:t['sha256']=hashlib.sha256(t['text'].encode()).hexdigest()
(ROOT/'texts.json').write_text(json.dumps(texts,ensure_ascii=False,indent=2))
manifest={'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'texts_file_sha256':hashlib.sha256((ROOT/'texts.json').read_bytes()).hexdigest(),'counts':dict(collections.Counter(t['category'] for t in texts)),'note':'All texts selected/generated before any detector test; no outcome-based sample removal or replacement.'}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('FROZEN',manifest,flush=True)
