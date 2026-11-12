import pathlib,json,hashlib,datetime,collections
R=pathlib.Path(__file__).resolve().parent
assert not (R/'results.json').exists(), 'Never refreeze after detector results'
x=json.loads((R/'texts.json').read_text());now=datetime.datetime.now(datetime.timezone.utc).isoformat()
# Source-only genre audit, before testing. Some letter-selection buckets span actual genres.
updates={'R2H005':('work_email','formal'),'R2H007':('reference_request','formal'),'R2H009':('inquiry_letter','formal'),'R2H010':('cover_letter','formal'),'R2H012':('cover_letter','formal'),'R2H014':('cover_letter','formal'),'R2H015':('work_email','casual'),'R2H019':('essay','formal'),'R2H020':('review','formal'),'R2H021':('short_story','casual'),'R2H022':('essay','formal'),'R2H023':('essay','formal'),'R2H024':('application_answer','formal')}
oldhash={hashlib.sha256(t['text'].encode()).hexdigest() for t in json.loads((R.parent/'round-1/texts.json').read_text())}
for t in x:
 if t['id'] in updates:t['genre'],t['register']=updates[t['id']]
 t['frozen_at']=now;t['word_count']=len(t['text'].split());t['sha256']=hashlib.sha256(t['text'].encode()).hexdigest();assert t['sha256'] not in oldhash;assert 60<=t['word_count']<=500
assert len(x)==120 and len({t['sha256'] for t in x})==120
(R/'texts.json').write_text(json.dumps(x,ensure_ascii=False,indent=2))
m=dict(frozen_at=now,texts_file_sha256=hashlib.sha256((R/'texts.json').read_bytes()).hexdigest(),counts=dict(collections.Counter(t['category'] for t in x)),word_range=[min(t['word_count'] for t in x),max(t['word_count'] for t in x)],note='Final source-only annotation audit and timestamp before any detector call. No outcome-based selection or text editing. No round-1 texts reused.')
(R/'manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2),flush=True)
