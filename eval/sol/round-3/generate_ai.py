import pathlib,json,urllib.request,urllib.error,time,re,datetime,hashlib
import yaml
R=pathlib.Path(__file__).resolve().parent
C=yaml.safe_load((pathlib.Path.home()/'docker/cliproxyapi/config.yaml').read_text());KEY=C.get('api-keys',C.get('access',{}).get('api-keys'))[0]
MODELS=['gpt-6.1-terra','gemini-3.1-flash-lite','claude-sonnet-4-6']
H=json.loads((R/'humans.json').read_text());O=R/'generation';O.mkdir(exist_ok=True)
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
profiles=[
'Fictional applicant: repair-desk assistant learning SQL, handled 40 repair tickets a week and built a spreadsheet for missing charger returns; seeking junior operations work at an invented bicycle-rental firm called Birch Wheel. Old colleague named Robin.',
'Fictional applicant: evening library assistant, maintained room bookings and helped with a move of 300 reference books; applying to a junior project support role at a fictional housing charity called Cedar Rooms. Old colleague named Alex.',
'Fictional applicant: laboratory stores assistant, checked deliveries against purchase orders and tracked stock of gloves; seeking entry-level data support at fictional water-testing firm called Reed Sample. Old colleague named Sam.'
]
A=[]
def gen(model,tag,specs,profile):
 path=O/(model+'-'+tag+'.json')
 if path.exists():return json.loads(path.read_text())['parsed']
 prompt=('Write independent synthetic examples for a private AI-text-detector evaluation. These are fictional messages, never genuine applications. Return ONLY a valid JSON array of objects with id and text, one per specification. No commentary or markdown fences. Do not mention this evaluation in the writing. Different voices and rhythms; do not recycle distinctive phrases between examples. Do not include names/addresses/emails of real people, hyperlinks, placeholders or credentials. Invent missing details. Short specs MUST contain 30–60 whitespace-separated words, preferably 40–55. Normal specs should contain 85–170 words, unless editing a longer original. A restrained voice is ordinary rather than a performance: no fake typos unless requested, no forced slang, no motivational finale. Preserve every fact of supplied editing sources.\nPROFILE: '+profile+'\nSPECS:\n'+json.dumps(specs,ensure_ascii=False))
 body={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0.85,'max_tokens':6500}
 for attempt in range(2):
  started=now();tic=time.monotonic()
  try:
   q=urllib.request.Request('http://127.0.0.1:8317/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
   with urllib.request.urlopen(q,timeout=600) as response:rawbody=response.read().decode();status=response.status
   # Record every raw reply before validating; no outcome-driven regeneration.
   raw=json.loads(rawbody);(O/(model+'-'+tag+f'-attempt-{attempt+1}.json')).write_text(json.dumps(dict(request=body,raw_http_body=rawbody,response=raw,http_status=status,started_at=started,elapsed_seconds=time.monotonic()-tic),ensure_ascii=False,indent=2))
   content=raw['choices'][0]['message']['content'];content=re.sub(r'^```(?:json)?\s*|\s*```$','',content.strip());got=json.loads(content)
   assert isinstance(got,list) and len(got)==len(specs)
   byid={x['id']:x for x in got};got=[byid[x['id']] for x in specs]
   for spec,x in zip(specs,got):
    n=len(x['text'].split());assert 30<=n<=500,(x['id'],n)
    # Retain valid prose even if requested short/normal length is missed; annotate the actual slice, never truncate it.
   result=dict(model=model,tag=tag,request=body,response=raw,raw_http_body=rawbody,parsed=got,generated_at=now(),elapsed_seconds=time.monotonic()-tic)
   path.write_text(json.dumps(result,ensure_ascii=False,indent=2));print('GENERATED',model,tag,[(x['id'],len(x['text'].split())) for x in got],flush=True);return got
  except Exception as e:
   print('GENERATION ERROR',model,tag,attempt+1,repr(e),flush=True)
   err=dict(started_at=started,error=repr(e))
   if isinstance(e,urllib.error.HTTPError):err.update(http_status=e.code,response_body=e.read().decode(errors='replace'))
   (O/(model+'-'+tag+f'-error-{attempt+1}.json')).write_text(json.dumps(err,indent=2))
   if attempt==1:raise
   time.sleep(20)

def collect(model,tag,specs,cat,profile):
 got=gen(model,tag,specs,profile)
 for s,x in zip(specs,got):
  t=x['text'].strip();n=len(t.split());p=s.get('priority_genre')
  row=dict(id=x['id'],label='ai_edited' if cat=='ai_polished_human' else 'ai',category=cat,genre=s['genre'],priority_genre=p,priority=bool(p),formality=s.get('formality','casual_professional' if p else 'formal'),subtype=s.get('subtype','raw'),text=t,word_count=n,short_30_60=30<=n<=60,source=dict(model=model,generation_file=f'generation/{model}-{tag}.json',prompt_spec=s,provenance_note='Requested proxy model identifier; underlying identity/account not independently authenticated.'))
  if 'parent_id' in s:row['parent_id']=s['parent_id']
  A.append(row)

for mi,model in enumerate(MODELS):
 prefix=f'R3M{mi+1}';profile=profiles[mi]
 raw=[
 ('linkedin_outreach','Introduce yourself to a team lead after reading about a junior vacancy. Ask one specific question about the daily work, no request for a call.',True),
 ('recruiter_reply','Reply that you are interested but need to know whether the job includes Saturday shifts before booking an interview.',True),
 ('application_answer','Answer: describe a small process you improved. Use one concrete detail and a limitation; do not invent management experience.',False),
 ('linkedin_post','Write a modest post about finishing a first useful work spreadsheet, an annoying error and the correction. Not a list of lessons.',False),
 ('scientific_encyclopedic','Explain what a floodplain does when a river rises and the limits of flood storage; no employment story.',False),
 ('short_story','Write a short scene about somebody returning a borrowed saucepan with a lid that does not fit. Restrained dialogue, no moral.',False)
 ]
 specs=[]
 for i,(genre,task,short) in enumerate(raw):specs.append(dict(id=f'{prefix}R{i+1:02}',genre=genre,priority_genre=genre if i<4 else None,length='short' if short else 'normal',task=task+' Use ordinary clear professional prose.' if i<4 else task))
 rawspecs=specs;collect(model,'raw',specs,'ai_raw',profile);rawrows=A[-6:]
 hard=[
 ('networking_referral_ask','natural_restrained','Ask an old colleague whether a vacancy is actually entry-level before deciding to apply. Do not pretend to be close; one shared detail.',True),
 ('follow_up','natural_restrained','Follow up nine days after sending a work sample. Mention the particular file and ask whether anything is missing, not for a decision.',True),
 ('thank_you_note','natural_restrained','Thank an interviewer for explaining one mundane part of the job. Mention what you still need to learn, without a polished conclusion.',True),
 ('application_answer','natural_restrained','Answer why the job interests you. The real reason is steadier hours and work you recognize; make that sound reasonable, not inspirational.',False),
 ('recruiter_reply','natural_restrained','Correct a recruiter who seems to have mistaken your background for more senior experience. Be interested but plain about the gap.',False),
 ('linkedin_outreach','natural_restrained','A second-degree connection offered to answer questions after a local talk. Ask about one tool used in the team. No flattery, balanced three-part structure or grand claims.',False),
 ('linkedin_post','lowercase','An all-lowercase update about changing a small work habit after forgetting a handover detail. Be understated, no emojis or hashtags.',True),
 ('networking_referral_ask','non_native_imitation','Intermediate-English speaker asking if a former colleague could refer them for a junior job. A few believable article or tense errors, no caricature of an L1.',False),
 ('application_answer','humanize_paraphrase','Rewrite the supplied AI application answer into restrained, specific casual-professional prose. Keep all facts, remove polished framing; do not deliberately insert mistakes.',False),
 ('follow_up','humanize_paraphrase','Rewrite the supplied AI draft into an unshowy follow-up. Preserve facts and the ask. Vary sentence lengths without stylized tricks.',False)
 ]
 allspec=[]
 for i,(genre,sub,task,short) in enumerate(hard):
  s=dict(id=f'{prefix}H{i+1:02}',genre=genre,priority_genre=genre,subtype=sub,length='short' if short else 'normal',task=task)
  if i==8:s.update(parent_id=rawrows[2]['id'],source_text=rawrows[2]['text'])
  if i==9:s.update(source_text='I sent the work sample named stock-check.xlsx nine days ago after our interview. It contains a sheet that matches incoming deliveries to purchase orders. Could you please let me know whether you received it and whether I should send anything else? I understand that the team may still be reviewing applications.',parent_kind='fixed reviewer-written AI draft')
  allspec.append(s)
 for start in [0,5]:collect(model,f'hard-{start}',allspec[start:start+5],'ai_casual_or_humanized',profile)
 parents=[['R3H002','R3H009','R3H042','R3H065'],['R3H005','R3H011','R3H043','R3H074'],['R3H006','R3H010','R3H041','R3H046']][mi]
 specs=[]
 for i,hid in enumerate(parents):
  h=next(h for h in H if h['id']==hid);specs.append(dict(id=f'{prefix}P{i+1:02}',genre=h['genre'],priority_genre=h['priority_genre'],subtype='polished_human',parent_id=hid,source_text=h['text'],length='short' if h['short_30_60'] else 'normal',task='Polish this public pre-2022 human text lightly for clarity and fluency. Preserve all claims and meaning; no invented details. Keep roughly its original length. Do not turn it into a new fictional story.'))
 if mi==1:
  collect(model,'polished-source-only',specs,'ai_polished_human','NO FICTIONAL PROFILE APPLIES. Each source_text is the sole authority. You are a copy editor, not an applicant. Never replace jobs, institutions, names, science, historical facts or the original purpose with a different scenario. Preserve every claim, including claims you would not personally make. No extraneous explanation.')
 else:collect(model,'polished',specs,'ai_polished_human',profile)
(R/'proxy_ai.json').write_text(json.dumps(A,ensure_ascii=False,indent=2));assert len(A)==60
print('PROXY GENERATION COMPLETE',now(),flush=True)
