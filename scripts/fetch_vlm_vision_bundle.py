"""Extract only the official vision context and sample tensors with HTTP ranges."""
from pathlib import Path
import json,requests,struct,zlib,hashlib,shutil
ROOT=Path('/home/youngmin/dev/llm/QNN_Exporter')
OUT=ROOT/'artifacts/vlm-vision-backends-20261008/official'
INDEX=Path('/home/youngmin/dev/llm/MobiSpec.cpp/docs/results/raw/vlm-vision-backends-20261008/official-bundle-index.json')
r=json.loads(INDEX.read_text());OUT.mkdir(parents=True,exist_ok=True)
def range_bytes(start,end):
 response=requests.get(r['url'],headers={'Range':f'bytes={start}-{end}'},timeout=120)
 response.raise_for_status();assert response.status_code==206;assert len(response.content)==end-start+1
 return response.content
manifest={}
for i in r['entries']:
 name=Path(i['name']).name
 if not(i['name'].endswith('/vision_encoder.bin') or '/sample_inputs/' in i['name'] and name.endswith('.raw')):continue
 dest=OUT/name
 if dest.exists():
  assert dest.stat().st_size==i['size'];manifest[name]={'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':i['size']};continue
 assert shutil.disk_usage(OUT).free-i['size']>50*(1<<30),'Host storage guard'
 header=range_bytes(i['offset'],i['offset']+29);assert header[:4]==b'PK\x03\x04'
 filename,extra=struct.unpack_from('<HH',header,26);start=i['offset']+30+filename+extra
 dec=zlib.decompressobj(-15) if i['compression']==8 else None
 digest=hashlib.sha256();count=0;received=0;tmp=dest.with_suffix(dest.suffix+'.partial')
 with requests.get(r['url'],headers={'Range':f'bytes={start}-{start+i["compressed_size"]-1}'},timeout=120,stream=True) as response:
  response.raise_for_status();assert response.status_code==206
  with tmp.open('wb') as f:
   for chunk in response.iter_content(1<<20):
    received+=len(chunk);data=dec.decompress(chunk) if dec else chunk;f.write(data);digest.update(data);count+=len(data)
   if dec:
    data=dec.flush();f.write(data);digest.update(data);count+=len(data);assert dec.eof
 assert received==i['compressed_size'] and count==i['size'],(received,count,i)
 tmp.replace(dest);manifest[name]={'sha256':digest.hexdigest(),'bytes':count}
 print('EXTRACTED',name,count,flush=True)
(OUT/'download-manifest.json').write_text(json.dumps(dict(source=r['url'],zip_bytes=r['size'],files=manifest),indent=2)+'\n')
print('VISION_ONLY_DOWNLOAD_COMPLETE',flush=True)
