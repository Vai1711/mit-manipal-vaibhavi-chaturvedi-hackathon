"""Explicit, resumable-by-file download of pinned safe model files. No training."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--include-reference',action='store_true',help='Also download the optional DistilRoBERTa benchmark reference');args=parser.parse_args()
    specs=json.loads((ROOT/'data/model_specs.json').read_text(encoding='utf-8'))
    for role,spec in specs.items():
        if role=='sentiment' and not args.include_reference:continue
        folder=ROOT/spec['local_directory'];folder.mkdir(parents=True,exist_ok=True)
        for info in spec['files']:
            dest=folder/info['name']
            if dest.exists() and (not info['sha256'] or hashlib.sha256(dest.read_bytes()).hexdigest()==info['sha256']):continue
            url=f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{info['name']}"
            temp=dest.with_suffix(dest.suffix+'.part')
            print('Downloading',role,info['name'],flush=True)
            request=urllib.request.Request(url,headers={'User-Agent':'RiskRadarHackathon/2.0'})
            digest=hashlib.sha256()
            with urllib.request.urlopen(request,timeout=90) as response,temp.open('wb') as output:
                while chunk:=response.read(1024*1024):output.write(chunk);digest.update(chunk)
            if info['sha256'] and digest.hexdigest()!=info['sha256']:
                temp.unlink(missing_ok=True);raise ValueError('Model checksum mismatch: '+info['name'])
            temp.replace(dest)
    print('Pinned models ready. No training required.')

if __name__=='__main__':main()
