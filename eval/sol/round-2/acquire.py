import pathlib,urllib.request,json,time,tarfile
R=pathlib.Path(__file__).resolve().parent
sources={'wi':'https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz','tea':'https://www.gutenberg.org/cache/epub/769/pg769.txt','austen':'https://www.gutenberg.org/cache/epub/1342/pg1342.txt','darwin':'https://www.gutenberg.org/cache/epub/1228/pg1228.txt'}
for name,url in sources.items():
 d=R/'downloads'/name;d.mkdir(parents=True,exist_ok=False)
 try:
  b=urllib.request.urlopen(url,timeout=180).read();(d/('source.tar.gz' if name=='wi' else 'source.txt')).write_bytes(b);print(name,len(b),flush=True)
  if name=='wi':
   with tarfile.open(d/'source.tar.gz') as t:
    print(t.getnames()[:30],flush=True)
    for f in t.getmembers():
     if f.name.endswith('.json'):print(f.name,t.extractfile(f).read(2500).decode(),flush=True);break
 except Exception as e:print(name,repr(e),flush=True)
for site in ['interpersonal','opensource','softwareengineering']:
 d=R/'downloads'/site;d.mkdir(parents=True,exist_ok=False)
 url=f'https://api.stackexchange.com/2.3/questions?site={site}&pagesize=100&sort=creation&order=desc&todate=1546300799&filter=withbody'
 try:
  b=urllib.request.urlopen(url,timeout=90).read();(d/'questions.json').write_bytes(b);j=json.loads(b);print(site,len(j.get('items',[])),j.get('error_message',''),flush=True)
 except Exception as e:print(site,repr(e),flush=True)
 time.sleep(1)
import yaml
c=yaml.safe_load((pathlib.Path.home()/'docker/cliproxyapi/config.yaml').read_text());k=c.get('api-keys',c.get('access',{}).get('api-keys'))[0]
req=urllib.request.Request('http://127.0.0.1:8317/v1/models',headers={'Authorization':'Bearer '+k})
j=json.load(urllib.request.urlopen(req,timeout=30));(R/'models.json').write_text(json.dumps(j,indent=2));print('MODELS',[x['id'] for x in j['data']],flush=True)
