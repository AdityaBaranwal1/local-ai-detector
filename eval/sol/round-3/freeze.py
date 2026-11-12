import pathlib,json,hashlib,datetime,collections,re,os
R=pathlib.Path(__file__).resolve().parent
assert not (R/'texts.json').exists() and not (R/'manifest.json').exists(),'Refuse to overwrite a frozen set'
H=json.loads((R/'humans.json').read_text());A=json.loads((R/'proxy_ai.json').read_text());S=json.loads((R/'self_ai.json').read_text());now=datetime.datetime.now(datetime.timezone.utc).isoformat()
for i,x in enumerate(S):
 x.update(id=f'R3S{i+1:03}',label='ai_edited' if x['category']=='ai_polished_human' else 'ai',priority=bool(x['priority_genre']),formality='casual_professional' if x['priority_genre'] else 'formal',word_count=len(x['text'].split()),source=dict(model='gpt-6.1-sol',method='Written directly by independent reviewer before detector calls',generated_at=now,provenance_note='Session model label supplied by harness; not a proxy call. Synthetic facts and people.'))
 x['short_30_60']=30<=x['word_count']<=60
 if 'parent_id' in x:
  parent=next(h for h in H if h['id']==x['parent_id']);x['source']['human_parent_sha256']=hashlib.sha256(parent['text'].encode()).hexdigest()
# Exclude one publisher-anonymized signature from both sides of its polishing pair.
for x in H:
 if x['id'] in ('R3H010','R3H011'):
  original=x['text'];assert '\nSincerely,' in original
  x['text']=original.split('\nSincerely,')[0].strip();x['word_count']=len(x['text'].split());x['short_30_60']=30<=x['word_count']<=60
  x['source']['char_end']=x['source']['char_start']+len(x['text']);x['source']['signature_omitted']=True
for x in A:
 if x['id']=='R3M2P02':
  original=x['text'];assert '\nSincerely,' in original
  x['text']=original.split('\nSincerely,')[0].strip();x['word_count']=len(x['text'].split());x['short_30_60']=30<=x['word_count']<=60
  x['source']['reviewer_postedit']={'operation':'Omit signature starting at Sincerely, to match the human body-only excerpt. Original source and generated full text remain in the saved batch.','reason':'Publisher-anonymized signature is not an authorship signal. Before any detector call.','reviewer':'gpt-6.1-sol'}
 if x['id']=='R3M3H06':
  assert x['text'].count('[name]')==1
  x['source']['reviewer_postedit']={'before':'[name]','after':'Jordan','reason':'Replace one leftover placeholder with a fictional addressee before freeze; no detector feedback.','reviewer':'gpt-6.1-sol'}
  x['text']=x['text'].replace('[name]','Jordan');x['word_count']=len(x['text'].split());x['short_30_60']=30<=x['word_count']<=60
T=H+A+S
expected=dict(human_native=40,human_non_native=40,ai_raw=25,ai_casual_or_humanized=40,ai_polished_human=15)
assert len(T)==160 and dict(collections.Counter(x['category'] for x in T))==expected
old=[]
for n in [1,2]:old+=json.loads((R.parent/f'round-{n}/texts.json').read_text())
norm=lambda t:re.sub(r'\W+',' ',t.lower()).strip()
oldhashes={hashlib.sha256(x['text'].encode()).hexdigest() for x in old};oldnorm={norm(x['text']) for x in old}
for x in T:
 assert 30<=x['word_count']<=500 and x['word_count']==len(x['text'].split()),(x['id'],x['word_count'])
 x['sha256']=hashlib.sha256(x['text'].encode()).hexdigest();x['frozen_at']=now
 assert x['sha256'] not in oldhashes and norm(x['text']) not in oldnorm,x['id']
 assert not re.search(r'ANON_|https?://|[\w.+-]+@[\w.-]+|\[(?:last name|first name|name|company)\]',x['text'],re.I),x['id']
 if x['label']=='human':
  assert x['source'].get('publication_year',2000)<2022
  if 'created_date' in x['source']:assert int(x['source']['created_date'][:4])<2022
assert len({x['sha256'] for x in T})==160 and len({norm(x['text']) for x in T})==160
assert sum(x['priority'] for x in T)>=54
payload=json.dumps(T,ensure_ascii=False,indent=2)+'\n';(R/'texts.json').write_text(payload)
manifest=dict(round=3,phase='A',frozen_at=now,texts_file='texts.json',texts_file_sha256=hashlib.sha256(payload.encode()).hexdigest(),sample_count=len(T),counts=expected,word_range=[min(x['word_count'] for x in T),max(x['word_count'] for x in T)],priority_count=sum(x['priority'] for x in T),priority_genres=dict(collections.Counter(x['priority_genre'] for x in T if x['priority'])),short_30_60_count=sum(x['short_30_60'] for x in T),short_counts=dict(collections.Counter(x['category'] for x in T if x['short_30_60'])),earlier_rounds=[dict(round=n,texts_file_sha256=hashlib.sha256((R.parent/f'round-{n}/texts.json').read_bytes()).hexdigest()) for n in [1,2]],deduplication='All 160 distinct by exact SHA-256 and lowercased alphanumeric/whitespace normalization; no exact/normalized overlap with 240 texts in rounds 1–2.',thresholds=dict(zerogpt_percent=50,ours_phase_B=0.5),local_service_called=False,selection_note='All selections, exclusions, annotations and generations predate the first detector call. Frozen texts.json must never be changed. No private owner text included.',limitations=['Native/proficient proxy does not establish individual forum authors’ L1.','Human priority examples are classroom application answers and thanks plus historical actual correspondence; none is verified modern LinkedIn/recruiter correspondence.','Human non-native examples share one collection, although authors and recorded L1s vary.','Different requested generator IDs do not establish model-family independence or training absence.'])
(R/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
# Files remain readable by phase B but refuse accidental in-place writes.
os.chmod(R/'texts.json',0o444);os.chmod(R/'manifest.json',0o444)
print(json.dumps(manifest,indent=2),flush=True)
