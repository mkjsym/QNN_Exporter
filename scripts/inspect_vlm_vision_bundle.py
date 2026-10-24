from pathlib import Path
import requests,re,json,io,zipfile,sys
raw=Path('/home/youngmin/dev/llm/MobiSpec.cpp/docs/results/raw/vlm-vision-backends-20261008')
url='https://huggingface.co/qualcomm/Qwen3-VL-4B-Instruct/raw/main/README.md'
r=requests.get(url,timeout=60);r.raise_for_status();(raw/'official-model-card.md').write_text(r.text)
links=re.findall(r'\[Download\]\(([^)]+)\)',r.text)
print('DOWNLOAD_URLS',json.dumps(links),flush=True)
url=next(u for u in links if 'geniex_qairt' in u.lower() and ('gen5' in u.lower() or '8elitegen5' in u.lower() or 'sm8850' in u.lower()))
class RemoteZip(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.pos=0;h=requests.head(url,allow_redirects=True,timeout=60);h.raise_for_status();self.size=int(h.headers['Content-Length']);self.headers=dict(h.headers)
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('negative offset')
        return self.pos
    def read(self,size=-1):
        if size<0:size=self.size-self.pos
        size=min(size,self.size-self.pos)
        if not size:return b''
        assert size<=2*(1<<20),size
        with requests.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+size-1}'},timeout=120,stream=True) as r:
            assert r.status_code==206,(r.status_code,dict(r.headers))
            data=r.content;assert len(data)==size,(len(data),size);self.pos+=len(data);return data
remote=RemoteZip(url)
with zipfile.ZipFile(remote) as z:
    entries=[dict(name=i.filename,size=i.file_size,compressed_size=i.compress_size,offset=i.header_offset,compression=i.compress_type) for i in z.infolist()]
    (raw/'official-bundle-index.json').write_text(json.dumps(dict(url=url,size=remote.size,headers=remote.headers,entries=entries),indent=2)+'\n')
    print('BUNDLE',remote.size,'bytes',len(entries),'entries',flush=True)
    for i in entries:
        if any(t in i['name'].lower() for t in ['vision','image','encoder','metadata','.so','.json']):print(i,flush=True)
    for i in z.infolist():
        if i.filename.endswith('.json') and i.file_size<2*(1<<20):
            dest=raw/'official-metadata'/i.filename;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(i));print('METADATA',i.filename,flush=True)
