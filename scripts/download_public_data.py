"""Explicit retrieval of exact public dataset revisions packaged here."""
import json,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    folder=ROOT/'data/public'
    metadata=json.loads((folder/'provenance.json').read_text())
    for spec in metadata.values():
        for name,expected in spec['files'].items():
            url='https://huggingface.co/datasets/'+spec['repository']+'/resolve/'+spec['revision']+'/'+name
            with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
            if hashlib.sha256(data).hexdigest()!=expected['sha256']:raise ValueError('Dataset hash mismatch: '+name)
            temp=folder/(name+'.part');temp.write_bytes(data);temp.replace(folder/name)
            print('Verified',name)
if __name__=='__main__':main()
