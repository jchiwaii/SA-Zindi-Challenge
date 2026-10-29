"""Numerical checks for the actual target, missingness, band conversion and splits."""
import sys
import io
import numpy as np
import pandas as pd
from acoustic_model import energy,features
from simulation import Simulator,BASE

def main():
    # 2 added units in 10 total units is .2; dB values must never be added directly.
    bg=np.full((3,15,25),8.)
    add=np.full_like(bg,2.)
    mask=np.ones_like(bg,dtype=bool);mask[1,:4]=False;mask[2,:,:4]=False
    target=(add*mask).sum((1,2))/((bg+add)*mask).sum((1,2))
    np.testing.assert_allclose(target,.2)
    db=np.where(mask,10*np.log10(bg+add),np.nan)
    e,offset=energy(db)
    np.testing.assert_allclose(e*10**(offset/10),np.where(mask,bg+add,0),rtol=1e-6)
    f,names=features(db)
    assert len(names)==f.shape[1] and len(names)==len(set(names))
    assert not np.isinf(f).any()
    # Adding 20 dB multiplies every energy by 100, but normalized fractions stay fixed.
    f2,_=features(db+20)
    ix=[i for i,s in enumerate(names) if s.startswith('band_energy_fraction')]
    np.testing.assert_allclose(f[:,ix],f2[:,ix],rtol=1e-6,atol=1e-6)
    from scipy.io import wavfile
    from prepare_external import minute_bands
    fs=32000
    wave=np.sin(2*np.pi*1000*np.arange(fs*60,dtype=np.float64)/fs).astype(np.float32)
    buf=io.BytesIO();wavfile.write(buf,fs,wave)
    converted,actual_fs=minute_bands(buf.getvalue())
    power=10**(converted/10)
    assert actual_fs==fs and converted.shape==(1,25)
    assert power[0,15]/power.sum()>.999,'1000 Hz tone must land in its correct band'
    m=pd.read_csv(BASE/'cache/background_windows.csv')
    assert m.window_id.is_unique
    assert (pd.to_datetime(m.start_export_clock).dt.year.between(2015,2025)).all()
    assert m.groupby(['recorder','time_block']).split.nunique().max()==1
    for recorder,g in m[~m.split.str.endswith('site')].groupby(['site','recorder']):
        tr=g[g.split=='train'].time_block;va=g[g.split=='val_time'].time_block;au=g[g.split=='audit_time'].time_block
        if len(tr) and len(va):assert tr.max()<va.min()
        if len(va) and len(au):assert va.max()<au.min()
    if (BASE/'cache/reference_manifest.csv').exists():
        r=pd.read_csv(BASE/'cache/reference_manifest.csv')
        assert r.groupby('hull_id').split.nunique().max()==1
        sim=Simulator();x,y,d=sim.generate(96,1234)
        assert np.isfinite(y).all() and ((y>=0)&(y<=1)).all()
        assert (y[~d.is_added]==0).all()
        x2,y2,_=sim.generate(96,1234)
        np.testing.assert_array_equal(x,x2);np.testing.assert_array_equal(y,y2)
    print('Passed: energy fractions, shared mask, gain invariance, safe transforms, chronological windows, and source isolation.')

if __name__=='__main__':main()
