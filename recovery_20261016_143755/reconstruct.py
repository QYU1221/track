import pathlib,json,re,shutil,hashlib
root=pathlib.Path.cwd();out=root/'recovery_20261016_143755';src=root/'space_target_detection'
if not (out/'current_backup').exists():shutil.copytree(src,out/'current_backup',ignore=shutil.ignore_patterns('__pycache__','datasets','*.dat'))
shutil.copytree(out/'current_backup',out/'restored',dirs_exist_ok=True)
patches=[]
for p in pathlib.Path(r'C:\Users\chenq\.codex\sessions\2026\10\16').glob('*.jsonl'):
 if '01a144fd' in p.name:continue
 for l in p.open(encoding='utf-8'):
  d=json.loads(l);q=d.get('payload',{});a=q.get('arguments',q.get('input',''))
  if d.get('timestamp','')<'2026-10-16T06:37:55' or q.get('name')!='exec':continue
  for m in re.finditer(r'tools\.apply_patch\(',a):
   try:patch,_=json.JSONDecoder().raw_decode(a[m.end():])
   except:continue
   patches.append((d['timestamp'],patch))
errors=[]
for ts,patch in sorted(patches,reverse=True):
 blocks=re.split(r'\*\*\* (Update|Add|Delete) File: ',patch)[1:]
 for i in range(0,len(blocks),2):
  op=blocks[i];name,_,body=blocks[i+1].partition('\n')
  if not name.startswith('space_target_detection/'):continue
  p=out/'restored'/name.split('/',1)[1]
  if op=='Add':
   if p.exists():p.unlink()
   continue
  if op!='Update' or not p.exists():errors.append((ts,name,'missing'));continue
  s=p.read_text(encoding='utf-8');lines=body.splitlines();hunks=[];h=[]
  for line in lines:
   if line.startswith('@@') or line.startswith('***'):
    if h:hunks.append(h);h=[]
   elif line[:1] in (' ','+','-'):h.append(line)
  if h:hunks.append(h)
  for h in reversed(hunks):
   new='\n'.join(x[1:] for x in h if x[0] in ' +');old='\n'.join(x[1:] for x in h if x[0] in ' -')
   if new and s.count(new)==1:s=s.replace(new,old,1)
   else:errors.append((ts,name,'unmatched',new[:120]))
  p.write_text(s,encoding='utf-8',newline='\n')
hashes=json.loads((root/'output/20261016_143755_647928_点目标筛选前1040.xlsx/tracking_summary.json').read_text())['source_sha256']
for n,h in hashes.items():
 p=out/'restored/v2'/n;b=p.read_bytes()
 if hashlib.sha256(b).hexdigest()!=h:
  if hashlib.sha256(b.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()==h:p.write_bytes(b.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
  else:print('MISMATCH',n)
print('ERRORS',json.dumps(errors,ensure_ascii=False));(out/'patches.json').write_text(json.dumps(sorted(patches),ensure_ascii=False),encoding='utf-8')

