"""Fetch a fixed hull-diverse public audio subset, verify, and convert locally.

Only archive byte ranges for preselected public training recordings are fetched.
Selection depends on public corpus metadata and duration, never competition inputs.
"""
from pathlib import Path
import io,json,hashlib,struct,zlib,time,zipfile,concurrent.futures
import numpy as np
import pandas as pd
import requests
from scipy.io import wavfile
from scipy.signal import periodogram

BASE=Path(__file__).resolve().parent
DEST=BASE/'external/vtuad_1.3.2'
TOTAL=3712337699
PART=1237445900

def remote_read(start,n):
    chunks=[]
    while n:
        part=start//PART;offset=start%PART;take=min(n,PART-offset)
        name=f'vtuad_strict_wav_vessel.zip.part{part+1:02d}'
        existing=DEST/(name+'.partial')
        if (DEST/name).exists():existing=DEST/name
        if existing.exists() and existing.stat().st_size>=offset+take:
            with existing.open('rb') as f:f.seek(offset);chunks.append(f.read(take))
        else:
            url=f'https://zenodo.org/api/records/23037895/files/{name}/content'
            for attempt in range(6):
                try:
                    with requests.get(url,headers={'Range':f'bytes={offset}-{offset+take-1}'},stream=True,timeout=(30,60)) as r:
                        r.raise_for_status()
                        if r.status_code!=206:raise ValueError('Server must honor byte ranges')
                        content=b''.join(r.iter_content(2**18))
                    if len(content)!=take:raise ValueError('Partial range')
                    chunks.append(content);break
                except Exception:
                    if attempt==5:raise
                    time.sleep(min(2**attempt,20))
        start+=take;n-=take
    return b''.join(chunks)

def minute_bands(wav_bytes):
    fs,samples=wavfile.read(io.BytesIO(wav_bytes))
    if samples.ndim>1:samples=samples[:,0]
    if np.issubdtype(samples.dtype,np.integer):
        samples=samples.astype(np.float32)/max(abs(np.iinfo(samples.dtype).min),np.iinfo(samples.dtype).max)
    samples=samples.astype(np.float32)
    nminutes=len(samples)//(60*fs)
    if nminutes<1:raise ValueError('Recording has no complete minute')
    centers=10**(1.5+.1*np.arange(25));result=[]
    for m in range(nminutes):
        segments=samples[m*60*fs:(m+1)*60*fs].reshape(30,2*fs)
        frequency,psd=periodogram(segments,fs=fs,window='hann',detrend='constant',scaling='density',axis=1)
        powers=np.full((30,25),np.nan)
        for j,c in enumerate(centers):
            low,high=c/10**.05,c*10**.05
            bins=(frequency>=low)&(frequency<high)
            if high<=fs/2 and bins.any():powers[:,j]=psd[:,bins].sum(1)*(frequency[1]-frequency[0])
        result.append(powers.mean(0))
    power=np.asarray(result)
    db=10*np.log10(np.maximum(power,1e-30))
    # Discard absolute digital gain. The next stage sets energy relative to SA backgrounds.
    db-=10*np.log10(power.mean())
    return db.astype(np.float32),fs

def convert(item):
    name,offset,compressed_size,row,checksum=item
    dest=DEST/'minute_bands'/f'{Path(name).stem}.npy'
    meta=dest.with_suffix('.json')
    if dest.exists() and meta.exists():return json.loads(meta.read_text())
    header=remote_read(offset,30)
    fields=struct.unpack('<4s5H3I2H',header)
    assert fields[0]==b'PK\x03\x04' and fields[3]==8
    start=offset+30+fields[-2]+fields[-1]
    compressed_path=DEST/'compressed_members'/f'{Path(name).stem}.deflate'
    if compressed_path.exists():compressed=compressed_path.read_bytes()
    else:
        compressed=remote_read(start,compressed_size)
        compressed_path.write_bytes(compressed)
    wav=zlib.decompress(compressed,-15)
    digest=hashlib.sha256(wav).hexdigest()
    assert digest==checksum, 'Public WAV checksum mismatch: '+name
    db,fs=minute_bands(wav)
    assert np.isfinite(db).all()
    np.save(dest,db)
    group=int(hashlib.sha256(('vessel-split-20261029|'+str(row['hull_id'])).encode()).hexdigest()[:8],16)%10
    split='val' if group in [0,1] else ('audit' if group==2 else 'train')
    result={**row,'archive_member':name,'strict_wav_sha256':digest,'feature_path':str(dest.relative_to(BASE)), 'complete_minutes':len(db),'actual_sample_rate':fs,'split':split}
    meta.write_text(json.dumps(result,indent=2))
    print(f'Converted {Path(name).name}: {len(db)} minutes, {row["label"]}, {split}',flush=True)
    return result

def main():
    for name in ['minute_bands','compressed_members']:(DEST/name).mkdir(parents=True,exist_ok=True)
    tail=DEST/'zip_tail.bin'
    if not tail.exists():tail.write_bytes(remote_read(TOTAL-65536,65536))
    index=DEST/'archive_index.sparse'
    if not index.exists():
        with index.open('wb') as f:f.seek(TOTAL-len(tail.read_bytes()));f.write(tail.read_bytes())
    with zipfile.ZipFile(index) as z:infos=z.infolist()
    by_number={Path(i.filename).stem:i for i in infos}
    registry=pd.read_csv(DEST/'registry_anonymized.csv')
    registry=registry[registry.label!='background'].copy()
    registry['archive_number']=registry.path.map(lambda x:Path(x).stem)
    registry['compressed_bytes']=registry.archive_number.map(lambda n:by_number[n].compress_size)
    # One shortest valid recording per public hull; add up to 8 per class for rare classes.
    chosen=registry.sort_values(['compressed_bytes','archive_number']).groupby('hull_id',sort=True).head(1)
    extras=registry.sort_values(['compressed_bytes','archive_number']).groupby('label',sort=True).head(8)
    chosen=pd.concat([chosen,extras]).drop_duplicates('archive_number').sort_values('archive_number',key=lambda s:s.astype(int))
    checksums={Path(line.split()[-1]).as_posix():line.split()[0] for line in (DEST/'strict_interval_wav_SHA256SUMS.txt').read_text().splitlines() if line.strip()}
    chosen.to_csv(DEST/'selected_recordings.csv',index=False)
    items=[]
    for row in chosen.to_dict('records'):
        inf=by_number[row['archive_number']]
        checksum=checksums.get(inf.filename)
        if checksum is None:checksum=next(v for k,v in checksums.items() if k.endswith('/vessel/'+row['archive_number']+'.wav'))
        items.append((inf.filename,inf.header_offset,inf.compress_size,row,checksum))
    print(f'Selected {len(items)} clips / {chosen.hull_id.nunique()} hulls / {chosen.compressed_bytes.sum()/1e6:.1f} MB',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(convert,items))
    pd.DataFrame(results).to_csv(BASE/'cache/reference_manifest.csv',index=False)
    print(pd.DataFrame(results).groupby(['split','label']).size().to_string(),flush=True)

if __name__=='__main__':main()
