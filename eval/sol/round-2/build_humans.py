import pathlib,json,tarfile,re,random,hashlib,collections
from html.parser import HTMLParser
R=pathlib.Path(__file__).resolve().parent;rng=random.Random(2026111202);texts=[]
class Plain(HTMLParser):
 def __init__(self):super().__init__();self.out=[]
 def handle_data(self,s):self.out.append(s)
 def handle_starttag(self,t,a):
  if t in ('p','br','li','blockquote'):self.out.append('\n')
 def handle_endtag(self,t):
  if t in ('p','li','blockquote'):self.out.append('\n')
def plain(s):
 p=Plain();p.feed(s);return re.sub(r'\n[ \t]*\n+','\n\n',''.join(p.out)).strip()
def add(text,cat,genre,register,source,native_status):
 wc=len(text.split());assert 60<=wc<=500,(wc,text[:80]);texts.append(dict(id=f'R2H{len(texts)+1:03}',label='human',category=cat,genre=genre,register=register,text=text,word_count=wc,source=source,native_status=native_status))
with tarfile.open(R/'downloads/wi/source.tar.gz') as t:
 docs=[]
 for level in ['A','B','C']:
  for line in t.extractfile(f'wi+locness/json/{level}.dev.json').read().decode().splitlines():
   d=json.loads(line);d['file']=f'wi+locness/json/{level}.dev.json';docs.append(d)
 used=set()
 def take(pred,n,genre,register):
  pool=[d for d in docs if pred(d) and 60<=len(d['text'].split())<=500 and not re.search(r'(?i)mother tongue is English|born in the UK|@|\d{6}',d['text'])];rng.shuffle(pool);got=0
  for d in pool:
   if d['userid'] in used:continue
   used.add(d['userid']);add(d['text'],'human_non_native',genre,register,dict(collection='Cambridge Write & Improve, original uncorrected ESL submissions, BEA2019 release 2.1',url='https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz',file=d['file'],document_id=d['id'],cefr=d['cefr'],publication_year=2019,privacy='Dataset manually anonymized by publishers; no personal identifiers retained'), 'ESL corpus provenance; individual L1 not released; no nationality inference');got+=1
   if got==n:break
  assert got==n,(genre,got)
 take(lambda d:bool(re.search(r'(?i)apply|applying|application|position|recruitment',d['text'])) and bool(re.search(r'(?i)^\s*(dear|hi |hello)',d['text'])),8,'cover_letter','formal')
 take(lambda d:bool(re.search(r'(?i)^\s*dear (sir|madam|mrs|mr)',d['text'])),4,'professional_letter','formal')
 take(lambda d:bool(re.search(r'(?i)^\s*(hi |hello|dear (jo|paul|jane|john|steve))',d['text'])),6,'casual_message','casual')
 take(lambda d:not re.search(r'(?i)^\s*(dear|hi |hello)',d['text']),6,'essay_review_or_story','formal')
 # Native student essays, different collection from round 1, no corrections.
 nd=[json.loads(x) for x in t.extractfile('wi+locness/json/N.dev.json').read().decode().splitlines()];rng.shuffle(nd)
 for d in nd[:8]:
  ps=re.split(r'\n\s*\n',d['text']);text=''
  for p in ps:
   if len((text+' '+p).split())>480:break
   text+=(('\n\n' if text else '')+p)
  if len(text.split())<60:
   # Long single paragraph: preserve first complete sentences up to 450 words.
   ss=re.split(r'(?<=[.!?])\s+',d['text']);parts=[]
   for s in ss:
    if len(' '.join(parts+[s]).split())>450:break
    parts.append(s)
   text=' '.join(parts)
  add(text,'human_native','academic_essay','formal',dict(collection='LOCNESS native English student essays, BEA2019 release',url='https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz',file='wi+locness/json/N.dev.json',document_id=d['id'],publication_year=2019,excerpt='opening whole paragraphs or complete sentences; uncorrected'), 'Native English at corpus level; individual L1 not released')
# Non-native literary English, three independently known first-language backgrounds.
def book(name,number,author,year,cat,genre,l1,n):
 s=(R/'downloads'/name/'source.txt').read_text(encoding='utf-8-sig');s=re.split(r'\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n',s,maxsplit=1)[1];s=re.split(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG',s)[0]
 ps=[re.sub(r'\s+',' ',p).strip() for p in re.split(r'\n\s*\n',s)];ps=[p for p in ps if 70<=len(p.split())<=400 and not re.search(r'Project Gutenberg|www\.|http|CONTENTS|Transcriber',p,re.I)]
 for frac in ([0.23,0.61] if n==2 else [0.18,0.43,0.71]):
  ix=int(len(ps)*frac);add(ps[ix],cat,genre,'formal',dict(collection=f'{author}, Project Gutenberg #{number}',url=f'https://www.gutenberg.org/ebooks/{number}',publication_year=year,paragraph_index_after_length_filter=ix,first_language=l1), 'Historical literary author; '+('non-English first language '+l1 if cat=='human_non_native' else 'native English author')+'; published/editorial prose, not unassisted learner writing')
book('tea',769,'Kakuzo Okakura',1906,'human_non_native','cultural_essay','Japanese',2)
book('tagore',6842,'Rabindranath Tagore',1913,'human_non_native','philosophical_essay','Bengali',2)
book('conrad',220,'Joseph Conrad',1910,'human_non_native','short_story','Polish',2)
# Public technical/forum posts: no actual application records or private correspondence.
for site,n in [('opensource',8),('softwareengineering',8)]:
 posts=json.loads((R/'downloads'/site/'questions.json').read_text())['items'];pool=[]
 for p in posts:
  if p['creation_date']>=1546300800 or p.get('last_edit_date',0)>=1640995200:continue
  text=plain(p['body']);wc=len(text.split())
  if not 60<=wc<=420 or re.search(r'@|https?://|\b(?:wife|husband|diagnos|suicid|salary|phone)\b|\d{7}',text,re.I):continue
  if '<code>' in p['body'] or '<pre>' in p['body']:continue
  pool.append((p,text))
 rng.shuffle(pool);assert len(pool)>=n,(site,len(pool))
 for p,text in pool[:n]:add(text,'human_native','technical_forum','casual',dict(collection=site+'.stackexchange.com',url=p['link'],question_id=p['question_id'],creation_date=p['creation_date'],last_edit_date=p.get('last_edit_date'),publication_year=2018), 'Unknown individual L1; English-language technical forum native/proficient proxy only')
book('darwin',1228,'Charles Darwin',1859,'human_native','scientific_encyclopedic','English',3)
book('austen',1342,'Jane Austen',1813,'human_native','short_story','English',3)
assert len(texts)==60,len(texts)
old=json.loads((R.parent/'round-1/texts.json').read_text());oldhash={hashlib.sha256(x['text'].encode()).hexdigest() for x in old}
for x in texts:assert hashlib.sha256(x['text'].encode()).hexdigest() not in oldhash
(R/'humans.json').write_text(json.dumps(texts,ensure_ascii=False,indent=2))
print(collections.Counter(x['category'] for x in texts));print(collections.Counter(x['genre'] for x in texts))
for x in texts:print(x['id'],x['word_count'],x['genre'],x['source'].get('document_id',x['source'].get('first_language',x['source'].get('question_id'))),repr(x['text'][:100]))
