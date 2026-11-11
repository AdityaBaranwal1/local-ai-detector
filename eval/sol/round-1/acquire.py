import urllib.request, json, pathlib, time
ROOT=pathlib.Path(__file__).resolve().parent
sources={
'fce': 'https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz',
'brown': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip',
}
for name,url in sources.items():
 d=ROOT/'downloads'/name;d.mkdir(parents=True,exist_ok=False)
 try:
  data=urllib.request.urlopen(url,timeout=180).read()
  (d/('source.tar.gz' if name=='fce' else 'source.zip')).write_bytes(data)
  print(name,len(data),flush=True)
 except Exception as e: print(name,repr(e),flush=True)
# Fetch dated, public prose, not answers to detector-related questions.
for site in ['workplace','academia','interpersonal','writing']:
 d=ROOT/'downloads'/site;d.mkdir(parents=True,exist_ok=False)
 url=f'https://api.stackexchange.com/2.3/questions?site={site}&pagesize=100&sort=creation&order=desc&todate=1577836799&filter=withbody'
 try:
  data=urllib.request.urlopen(url,timeout=90).read()
  (d/'questions.json').write_bytes(data)
  j=json.loads(data);print(site,len(j.get('items',[])),j.get('error_message',''),flush=True)
 except Exception as e:print(site,repr(e),flush=True)
 time.sleep(1)
# Inspect available proxy models without displaying any credential.
import yaml
config=yaml.safe_load(pathlib.Path('/home/you/docker/cliproxyapi/config.yaml').read_text())
keys=config.get('api-keys',config.get('access',{}).get('api-keys',[]))
if not keys:raise RuntimeError('No proxy key found')
r=urllib.request.Request('http://127.0.0.1:8317/v1/models',headers={'Authorization':'Bearer '+keys[0]})
j=json.load(urllib.request.urlopen(r,timeout=30))
print('Available models:',[x['id'] for x in j.get('data',[])],flush=True)
(ROOT/'models.json').write_text(json.dumps(j,indent=2))
