import pathlib,json,tarfile,zipfile,re,html,hashlib,random,collections
from html.parser import HTMLParser
ROOT=pathlib.Path(__file__).resolve().parent
rng=random.Random(1101126)
class Plain(HTMLParser):
 def __init__(self): super().__init__();self.out=[]
 def handle_data(self,s):self.out.append(s)
 def handle_starttag(self,tag,attrs):
  if tag in ('p','br','li','blockquote'):self.out.append('\n')
 def handle_endtag(self,tag):
  if tag in ('p','li','blockquote'):self.out.append('\n')
def plain(s):
 p=Plain();p.feed(s);return re.sub(r'\n[ \t]*\n+', '\n\n',''.join(p.out)).strip()
texts=[]
def add(text,category,genre,source,**meta):
 wc=len(text.split());assert 60<=wc<=500,(wc,text[:60])
 texts.append(dict(id=f'H{len(texts)+1:03}',label='human',category=category,genre=genre,text=text,word_count=wc,source=source,**meta))
# Original untokenized learner prose; never apply the supplied corrections.
with tarfile.open(ROOT/'downloads/fce/source.tar.gz') as t:
 docs=[]
 for split in ['test','dev','train']:
  for line in t.extractfile(f'fce/json/fce.{split}.json').read().decode().splitlines():
   d=json.loads(line);d['split']=split;docs.append(d)
# Select job/application letters first, other practical letters, then broader genres.
used_writers=set();selected=[]
def take(predicate,n,genre):
 pool=[d for d in docs if predicate(d) and 60<=len(d['text'].split())<=500];rng.shuffle(pool)
 count=0
 for d in pool:
  writer=d['id'].rsplit('*',1)[0]
  if writer in used_writers:continue
  used_writers.add(writer);selected.append((d,genre));count+=1
  if count==n:break
 assert count==n,(genre,count)
take(lambda d:bool(re.search(r'\b(apply|applying|application|advertisement|advertised|position|vacancy)\b',d['text'],re.I)) and bool(re.search(r'\b(Dear|Sir|Madam)\b',d['text'][:150],re.I)),8,'formal_complaint_or_letter')
take(lambda d:bool(re.search(r'\bDear\b',d['text'][:100],re.I)),12,'email_or_letter')
take(lambda d:not re.search(r'\bDear\b',d['text'][:100],re.I),10,'essay_review_or_story')
for d,g in selected:
 add(d['text'],'human_non_native',g,{'collection':'Cambridge FCE, original human exam scripts (2000–2001), published 2011; BEA2019 v2.1','url':'https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz','file':f"fce/json/fce.{d['split']}.json",'document_id':d['id'],'first_language':d['l1'],'publication_year':2011},native_status='documented non-English L1')
# Public forum prose: native status cannot be established; label native/proficient stratum honestly.
chosen={'workplace':[150484,150452,150399,150387,150381,150355,150339,150323],
'academia':[142165,142035,142066,142018]}
for site,ids in chosen.items():
 posts=json.loads((ROOT/'downloads'/site/'questions.json').read_text())['items']
 for qid in ids:
  p=next(p for p in posts if p['question_id']==qid)
  assert p['creation_date']<1640995200 and p.get('last_edit_date',0)<1640995200,(site,qid,'edited after cutoff')
  text=plain(p['body'])
  add(text,'human_native','forum_work_or_application',{'collection':site+'.stackexchange.com','url':p['link'],'question_id':qid,'title':html.unescape(p['title']),'creation_date':p['creation_date'],'last_edit_date':p.get('last_edit_date'),'publication_year':2019},native_status='unknown; English-language occupational forum, native/proficient proxy, not verified native')
# Public-domain human-written sample business letters, published in 1920s, not genuine applicant records.
s=(ROOT/'downloads/gutenberg/letters.html').read_text()
for section,indices,genres in [('application',[0,1,2,3,4],['cover_letter']*5),('reference',[0,4],['reference_request','recommendation_letter']),('inquiry',[1],['cold_outreach'])]:
 start=s.index(f'<a id="{section}">');end=s.find('<h3',start+20)
 part=s[start:end if end!=-1 else len(s)]
 ps=[plain(x) for x in re.findall(r'<p\b[^>]*>(.*?)</p>',part,re.S)]
 letters=[];cur=None
 for p in ps:
  p=re.sub(r'\[\d+\]','',p);p=re.sub(r'\s+',' ',p).strip()
  if re.match(r'(?i)^(?:my )?dear .{0,50}:$|^Gentlemen:$|^To Whom It May Concern:$',p):cur=[p]
  elif cur is not None:
   if re.match(r'(?i)^(?:very |yours |sincerely |cordially )*(?:truly|sincerely|yours)',p):
    letters.append('\n\n'.join(cur));cur=None
   else:cur.append(p)
 for ix,g in zip(indices,genres):
  add(letters[ix],'human_native',g,{'collection':'Mary Owens Crowther, How to Write Letters (1922/1923-era examples), Project Gutenberg #22222','url':'https://www.gutenberg.org/cache/epub/22222/pg22222-images.html#'+section,'section':section,'letter_index':ix,'publication_year':1923},native_status='published human-written American-English model letter; not a real applicant record')
assert len(texts)==50,len(texts)
# Brown is 1961 published American-English prose; strip POS tags only, detokenize punctuation.
with zipfile.ZipFile(ROOT/'downloads/brown/source.zip') as z:
 cats=dict(line.split() for line in z.read('brown/cats.txt').decode().splitlines())
 for cat,n in [('reviews',3),('belles_lettres',2),('fiction',3),('hobbies',2)]:
  files=[k for k,v in cats.items() if v==cat];rng.shuffle(files)
  for f in files[:n]:
   data=z.read('brown/'+f).decode();paragraphs=[];current=[]
   for line in data.splitlines():
    if not line.strip():
     if current:paragraphs.append(' '.join(current));current=[]
    else:current.extend(tok.rsplit('/',1)[0] for tok in line.split())
   if current:paragraphs.append(' '.join(current))
   # Fixed excerpt at the beginning: whole paragraphs, stop before 350 words.
   selected_p=[]
   for p in paragraphs:
    if len(' '.join(selected_p+[p]).split())>350 and selected_p:break
    selected_p.append(p)
    if len(' '.join(selected_p).split())>=180:break
   text='\n\n'.join(selected_p)
   text=re.sub(r'\s+([,.!?;:])',r'\1',text)
   text=text.replace('`` ','“').replace(" ''",'”').replace('` ',"'")
   add(text,'human_native',cat,{'collection':'Brown Corpus, 1961 American English; NLTK redistribution','url':'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip','file':'brown/'+f,'publication_year':1961},native_status='published American-English prose; corpus-level proxy, individual L1 not verified')
(ROOT/'humans.json').write_text(json.dumps(texts,ensure_ascii=False,indent=2))
print('Human set:',collections.Counter(t['genre'] for t in texts))
print('L1s:',collections.Counter(t['source'].get('first_language') for t in texts if t['category']=='human_non_native'))
print('Counts:',collections.Counter(t['category'] for t in texts))
for t in texts:print(t['id'],t['word_count'],t['genre'],t['source'].get('document_id',t['source'].get('question_id',t['source'].get('file'))),t['text'][:75].replace('\n',' '))
