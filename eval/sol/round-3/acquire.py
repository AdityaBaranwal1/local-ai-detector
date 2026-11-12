import pathlib, urllib.request, json, time
R=pathlib.Path(__file__).resolve().parent
S={
'pelic_answers':'https://media.githubusercontent.com/media/ELI-Data-Mining-Group/PELIC-dataset/master/corpus_files/answer.csv',
'pelic_questions':'https://media.githubusercontent.com/media/ELI-Data-Mining-Group/PELIC-dataset/master/corpus_files/question.csv',
'pelic_students':'https://media.githubusercontent.com/media/ELI-Data-Mining-Group/PELIC-dataset/master/corpus_files/student_information.csv',
'pelic_courses':'https://media.githubusercontent.com/media/ELI-Data-Mining-Group/PELIC-dataset/master/corpus_files/course.csv',
'muse_followup':'https://www.themuse.com/advice/5-job-search-tactics-you-must-stop-immediately',
'fast_recruiter':'https://www.fastcompany.com/40404577/the-only-five-recruiter-emails-your-job-search-will-ever-need',
'muse_network':'https://spectrumbrandscareers.wordpress.com/2015/08/11/the-secret-to-the-ask-an-email-template-you-wont-want-to-miss/',
'ucsf_network':'https://career.ucsf.edu/sites/g/files/tkssra15591/files/wysiwyg/ResearchersPDFInformationalInterviews.pdf',
}
for name,url in S.items():
 d=R/'downloads'/name;d.mkdir(parents=True,exist_ok=False)
 try:
  raw=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ResearchClient/1.0'}),timeout=150).read()
  ext='csv' if 'pelic' in name else ('pdf' if url.endswith('.pdf') else 'html')
  (d/('source.'+ext)).write_bytes(raw)
  (d/'metadata.json').write_text(json.dumps({'url':url,'bytes':len(raw)},indent=2));print(name,len(raw),flush=True)
 except Exception as e: print(name,repr(e),flush=True)
 time.sleep(1)
