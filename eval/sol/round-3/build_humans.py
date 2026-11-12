import pathlib,json,csv,re,hashlib,collections,random,html,urllib.request,time
R=pathlib.Path(__file__).resolve().parent
D=R/'downloads'
H=[]
def add(text,cat,genre,source,priority=None,formality='formal'):
 text=text.strip();n=len(text.split());assert 30<=n<=500,(genre,n)
 H.append(dict(id=f'R3H{len(H)+1:03}',label='human',category=cat,genre=genre,priority_genre=priority,priority=bool(priority),formality=formality,text=text,word_count=n,short_30_60=30<=n<=60,source=source))
# The collected corpus is 2005–2012, published in 2020. Keep original learner wording.
qs={x['question_id']:x for x in csv.DictReader((D/'pelic_questions/source.csv').open())}
st={x['anon_id']:x for x in csv.DictReader((D/'pelic_students/source.csv').open())}
co={x['course_id']:x for x in csv.DictReader((D/'pelic_courses/source.csv').open())}
rows=list(csv.DictReader((D/'pelic_answers/source.csv').open()))
used=set();langs=collections.Counter();questions=collections.Counter()
def safe(t):
 return not re.search(r'ANON_|https?://|www\.|[\w.+-]+@[\w.-]+|\d[\d ()+-]{7,}\d',t) and len(re.findall(r'[A-Za-z]',t))>90

def pelic(x,genre,priority=None,excerpt=None,formal='formal'):
 t=(excerpt if excerpt is not None else x['text']).strip();s=st[x['anon_id']];assert x['anon_id'] not in used and safe(t)
 source=dict(collection='PELIC v1.1',url='https://github.com/ELI-Data-Mining-Group/PELIC-dataset',citation='Juffs, Han & Naismith (2020), doi:10.5281/zenodo.3991977',license='CC BY-NC-ND 4.0; unchanged text or contiguous excerpt, non-commercial private evaluation',answer_id=x['answer_id'],question_id=x['question_id'],anonymous_author_id=x['anon_id'],created_date=x['created_date'],native_language=s['native_language'],level=co[x['course_id']]['level_id'],class_id=co[x['course_id']]['class_id'],assignment=qs[x['question_id']]['stem'],provenance_note='Genuine pre-2022 learner submission, not AI. Classroom task; not a verified actual applicant message.',original_sha256=hashlib.sha256(x['text'].encode()).hexdigest())
 if excerpt is not None:source.update(excerpt=True,char_start=x['text'].index(t),char_end=x['text'].index(t)+len(t))
 else:source['excerpt']=False
 add(t,'human_non_native',genre,source,priority,formal);used.add(x['anon_id']);langs[s['native_language']]+=1;questions[x['question_id']]+=1
# All seven different authors answering the explicitly application-related prompt.
for x in rows:
 if x['question_id']=='5243' and x['version']=='1' and safe(x['text']) and 60<len(x['text'].split())<=500:pelic(x,'application_answer','application_answer')
assert len(H)==7
# Five separate authors; retain a contiguous body if a publisher-anonymized greeting/signoff would otherwise enter the detector.
thanks=[x for x in rows if x['question_id']=='2497' and x['version']=='1']
thanks.sort(key=lambda x: ('ANON_' in x['text'],int(x['answer_id'])))
for x in thanks:
 if x['anon_id'] in used:continue
 t=x['text'].strip();excerpt=None
 if not safe(t):
  chunks=re.split(r'[^\n]*ANON_[^\n]*(?:\n|$)',x['text']);chunks=[z.strip() for z in chunks if safe(z) and 30<=len(z.split())<=500]
  if not chunks:continue
  excerpt=max(chunks,key=lambda z:len(z.split()))
  t=excerpt
 if 30<=len(t.split())<=500:pelic(x,'thank_you_note','thank_you_note',excerpt,formal='casual_professional')
 if sum(h['genre']=='thank_you_note' for h in H)==5:break
assert len(H)==12
# Independent authors and recorded L1s, writing classes only, excluding sentence drills and template/quoted passages.
rng=random.Random(2026111203);rng.shuffle(rows)
base=[]
for x in rows:
 if x['question_id'] not in qs or x['course_id'] not in co or x['anon_id'] not in st:continue
 t=x['text'].strip();q=qs[x['question_id']]['stem'];c=co[x['course_id']];lang=st[x['anon_id']]['native_language']
 if x['version']!='1' or c['class_id']!='w' or not safe(t) or not lang or lang.lower()=='english':continue
 if re.search(r'sentence with|each word|fill in|rewrite|capitalize|capital letters|copy|reading comprehension|paraphrase|summari[sz]|summary|choose.*correct',q,re.I):continue
 if t.count('?')>5 or re.search(r'(?im)^\s*(?:\d+\s*[.\-:)]|[A-E][.)]|[-*]\s|list\b)',t):continue
 if re.search(r'\[last name\]|\[first name\]|core vocabulary|\bvocabulary\s*:',t,re.I) or len(re.findall(r'[.!?](?:\s|$)',t))<2:continue
 if re.search(r'\bsentence\b|reading|make a list|summari|paraphrase|correct the|change the|re-?write|\b(adj|adv|noun|verb)\b',q,re.I):continue
 if not re.search(r'write|describ|discuss|do you|did you|what.*you|choose|your|think|essay|paragraph|draft|topic|why|opinion',q,re.I):continue
 tw=re.findall(r'\w+',t.lower());qw=re.findall(r'\w+',q.lower());qgrams={tuple(qw[j:j+6]) for j in range(len(qw)-5)}
 if sum(tuple(tw[j:j+6]) in qgrams for j in range(len(tw)-5))>max(4,len(tw)*0.25):continue
 base.append(x)
def pick(count,low,high,formal):
 n=0
 for x in base:
  lang=st[x['anon_id']]['native_language'];q=x['question_id']
  if x['anon_id'] in used or langs[lang]>=8 or questions[q]>=2 or not low<=len(x['text'].split())<=high:continue
  pelic(x,'learner_reflection' if formal=='casual' else 'learner_essay',formal=formal);n+=1
  if n==count:return
 raise RuntimeError(('Insufficient learner samples',n,count))
pick(10,30,60,'casual');pick(10,110,220,'formal');pick(8,221,420,'formal')
assert len(H)==40
# Historical letters: authentic published correspondence, with genre proxies explicitly identified.
s=(D/'stevenson/source.txt').read_text(encoding='utf-8-sig');paras=re.split(r'\n\s*\n',s)
chosen=[(573,'thank_you_note'),(734,'networking_referral_ask'),(996,'networking_referral_ask'),(1021,'follow_up'),(1039,'thank_you_note'),(1082,'follow_up'),(1133,'thank_you_note'),(1196,'follow_up'),(1241,'follow_up'),(1806,None),(1817,'thank_you_note'),(1957,'thank_you_note')]
for idx,p in chosen:
 t=' '.join(paras[idx].split());prior='\n\n'.join(paras[max(0,idx-6):idx]);dates=re.findall(r'\b(?:18|19)\d\d\b',prior)
 add(t,'human_native',p or 'work_letter',dict(collection='Robert Louis Stevenson, Letters, Volume 1',url='https://www.gutenberg.org/ebooks/622',publication_year=1899,author_died=1894,paragraph_index=idx,nearby_date_metadata=dates,excerpt=True,original_paragraph=paras[idx],provenance_note='Native-English historical author; genuine correspondence. Job-search genre analogy only: NOT a modern LinkedIn/recruiter message. Whitespace normalized; wording unaltered.'),p,'casual_professional')
# Fresh scientific and narrative authors/books; avoid round-1/2 authors.
for name,year,genre in [('huxley',1893,'scientific_encyclopedic'),('wells',1895,'short_story')]:
 text=(D/name/'source.txt').read_text(encoding='utf-8-sig');ps=re.split(r'\n\s*\n',text);eligible=[]
 for i,p in enumerate(ps):
  t=' '.join(p.split())
  if 100<=len(t.split())<=330 and safe(t) and i>50 and i<len(ps)-80 and not re.search(r'gutenberg|copyright|chapter|illustration|preface|contents|fig\.|footnote|http',t,re.I):eligible.append((i,t))
 # Spread positions through source, deterministic before outcomes.
 for i,t in [eligible[int((j+1)*len(eligible)/8)] for j in range(6)]:
  add(t,'human_native',genre,dict(collection='H. G. Wells, The Time Machine' if name=='wells' else 'T. H. Huxley, Darwiniana (Collected Essays, Volume II)',url='https://www.gutenberg.org/ebooks/'+('35' if name=='wells' else '6919'),publication_year=year,paragraph_index=i,excerpt=True,provenance_note='Native-English historical author, single original paragraph; whitespace normalized, not rewritten.'),formality='casual' if name=='wells' else 'formal')
# Modern informal/formal non-career forums from two sites not used in earlier rounds or builder exclusions.
for site in ['bicycles','diy']:
 d=D/('se_'+site);d.mkdir(exist_ok=True)
 f=d/'questions.json'
 if not f.exists():
  url=f'https://api.stackexchange.com/2.3/questions?site={site}&pagesize=100&sort=creation&order=desc&todate=1514764799&filter=withbody'
  raw=urllib.request.urlopen(url,timeout=90).read();f.write_bytes(raw);(d/'metadata.json').write_text(json.dumps({'url':url}))
 data=json.loads(f.read_bytes());assert 'items' in data,data
 eligible=[]
 for x in data['items']:
  body=x['body']
  if x.get('last_edit_date',0)>=1640995200 or re.search(r'<(?:pre|code|img)|href=|https?://',body,re.I):continue
  t=html.unescape(re.sub(r'<[^>]+>',' ',body));t=' '.join(t.split());n=len(t.split())
  if safe(t) and 30<=n<=400:
   eligible.append((x,t))
 short=[z for z in eligible if len(z[1].split())<=60];long=[z for z in eligible if 60<len(z[1].split())<=400]
 # 3+2 native short forums plus five short letters = 10 native short samples.
 ns=3 if site=='bicycles' else 2;selected=short[:ns]+long[:8-ns];assert len(selected)==8
 for x,t in selected:
  add(t,'human_native','practical_forum',dict(collection='Stack Exchange '+site,url=x['link'],question_id=x['question_id'],created_timestamp=x['creation_date'],last_edit_timestamp=x.get('last_edit_date'),excerpt=False,provenance_note='Original pre-2018 question, no post-2021 edit. HTML stripped; wording unchanged. Individual L1 unknown: English-proficient/native proxy only.'),formality='casual')
 time.sleep(1)
assert len(H)==80
assert len({hashlib.sha256(h['text'].encode()).hexdigest() for h in H})==80
(R/'humans.json').write_text(json.dumps(H,ensure_ascii=False,indent=2))
print('HUMANS',dict(collections.Counter(h['category'] for h in H)),flush=True)
print('L1',dict(langs),flush=True)
print('PRIORITY',dict(collections.Counter(h['priority_genre'] for h in H if h['priority'])),flush=True)
print('SHORT',dict(collections.Counter(h['category'] for h in H if h['short_30_60'])),flush=True)
for h in H:print(h['id'],h['word_count'],h['genre'],h['source']['collection'],h['source'].get('native_language'),repr(h['text'][:100]),flush=True)
