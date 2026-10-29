"""Audit only released training measurements; never read competition test values."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "solution" / "reports"
OUT.mkdir(parents=True, exist_ok=True)
BANDS = ["31.6", "39.8", "50.1", "63.1", "79.4", "100", "125.9", "158.5", "199.5", "251.2", "316.2", "398.1", "501.2", "631", "794.3", "1000", "1258.9", "1584.9", "1995.3", "2511.9", "3162.3", "3981.1", "5011.9", "6309.6", "7943.3"]
COLS = ["TOL_" + s for s in BANDS]

def main():
    reports = []
    for path in sorted((ROOT / "South_African_Real_Acoustics/acoustics").glob("*.parquet")):
        schema = pq.ParquetFile(path).schema.names
        df = pd.read_parquet(path, columns=[c for c in ["site", "recorder", "t", "dur"] + COLS if c in schema])
        for c in COLS:
            if c not in df: df[c] = np.nan
        a = df[COLS].to_numpy(dtype=np.float32)
        t = pd.to_datetime(df.t)
        row = {"file": path.name, "rows": len(df), "recorders": df.recorder.nunique(), "duration_quantiles": df.dur.quantile([0,.01,.5,.99,1]).to_dict(), "missing_fraction": float((~np.isfinite(a)).mean()), "finite_bands_per_row": dict(zip(*[x.tolist() for x in np.unique(np.isfinite(a).sum(1),return_counts=True)])), "level_quantiles": np.nanpercentile(a,[0,.01,1,5,50,95,99,99.99,100]).tolist(), "all_band_spread_below_6": int((np.nanmax(a,axis=1)-np.nanmin(a,axis=1)<6).sum()), "duplicate_recorder_minutes": int(pd.DataFrame({'r':df.recorder,'t':t.dt.floor('min')}).duplicated().sum()), "recorders_detail": []}
        for recorder, g in df.groupby('recorder'):
            vals=g[COLS].to_numpy(dtype=np.float32)
            ts=pd.to_datetime(g.t).sort_values()
            delta=ts.diff().dt.total_seconds()
            row['recorders_detail'].append({'recorder':recorder,'rows':len(g),'median_spectrum_db':np.nanmedian(vals,axis=0).tolist(),'missing_by_band':(~np.isfinite(vals)).mean(0).tolist(),'large_gaps':int((delta>90).sum()),'zero_or_negative_cells':int((vals<=0).sum()),'above_220_cells':int((vals>220).sum())})
        reports.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='recorders_detail'}),flush=True)
        del df,a
    (OUT/'training_audit.json').write_text(json.dumps(reports,indent=2))
    header=pd.read_csv(ROOT/'Test.csv',nrows=0).columns.tolist()
    expected=['ID']+[f'm{m:02d}_{c}' for m in range(1,16) for c in COLS]
    assert header==expected, 'Unexpected test schema'
    print('Test header only: expected 15 x 25 schema confirmed. No test acoustic values loaded.')

if __name__=='__main__': main()
