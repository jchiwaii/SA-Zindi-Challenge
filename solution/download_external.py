"""Download the public CC-BY-4.0 VTUAD v1.3.2 strict-interval corpus."""
from pathlib import Path
import concurrent.futures
import argparse
import hashlib
import json
import time
import requests

DEST = Path(__file__).resolve().parent / 'external' / 'vtuad_1.3.2'
RECORD = 'https://zenodo.org/api/records/23037895'
NAMES = ['dataset_card.md', 'CITATION.cff', 'license_evidence_v131.md',
         'registry_anonymized.csv', 'strict_interval_wav_SHA256SUMS.txt',
         'README_vtuad_strict_wav_vessel_parts.txt',
         'vtuad_strict_wav_vessel.zip.part01', 'vtuad_strict_wav_vessel.zip.part02',
         'vtuad_strict_wav_vessel.zip.part03']

def fetch(f):
    path = DEST / f['key']
    if path.exists() and path.stat().st_size == f['size']:
        h=hashlib.md5()
        with path.open('rb') as inp:
            for b in iter(lambda:inp.read(2**20),b''):h.update(b)
        digest = h.hexdigest()
        if f['checksum'] == 'md5:'+digest:
            print('Verified existing',path.name,flush=True); return
    partial = path.with_suffix(path.suffix+'.partial')
    for attempt in range(6):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            headers={'Range':f'bytes={offset}-'} if offset else {}
            with requests.get(f['links']['self'],headers=headers,stream=True,timeout=(30,90)) as r:
                r.raise_for_status()
                if offset and r.status_code!=206: offset=0
                mode='ab' if offset else 'wb'
                last=time.monotonic()
                with partial.open(mode) as out:
                    for block in r.iter_content(2**20):
                        out.write(block);offset+=len(block)
                        if time.monotonic()-last>25:
                            print(f'{path.name}: {offset/2**20:.0f}/{f["size"]/2**20:.0f} MiB',flush=True);last=time.monotonic()
            assert partial.stat().st_size==f['size']
            h=hashlib.md5()
            with partial.open('rb') as inp:
                for b in iter(lambda:inp.read(2**20),b''):h.update(b)
            assert f['checksum']=='md5:'+h.hexdigest(), 'Checksum mismatch'
            partial.rename(path)
            print('Downloaded and verified',path.name,flush=True);return
        except Exception as e:
            print('Retry',attempt,path.name,str(e),flush=True);time.sleep(min(2**attempt,30))
    raise RuntimeError('Download failed: '+path.name)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--metadata-only',action='store_true');args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    r=requests.get(RECORD,timeout=60);r.raise_for_status();d=r.json()
    assert d['metadata']['license']['id']=='cc-by-4.0'
    (DEST/'zenodo_record.json').write_text(json.dumps(d,indent=2))
    selected=[f for f in d['files'] if f['key'] in NAMES]
    assert len(selected)==len(NAMES)
    if args.metadata_only:selected=[f for f in selected if f['size']<1_000_000]
    print('Total requested GB',sum(f['size'] for f in selected)/1e9,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch,selected))
    (DEST/('metadata_download_complete.json' if args.metadata_only else 'download_complete.json')).write_text(json.dumps({'record':RECORD,'files':[f['key'] for f in selected],'version':'1.3.2','licence':'CC BY 4.0'},indent=2))

if __name__=='__main__':main()
