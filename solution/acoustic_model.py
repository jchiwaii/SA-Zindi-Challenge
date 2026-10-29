"""Fixed per-window acoustic transforms and train-only background templates."""
import warnings
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def energy(db):
    """Scale each window safely; NaN remains unobserved (zero only in sums)."""
    maximum=np.nanmax(db,axis=(1,2),keepdims=True)
    maximum=np.where(np.isfinite(maximum),maximum,0)
    e=np.where(np.isfinite(db),np.power(10.,(db-maximum)/10),0).astype(np.float32)
    return e,maximum

def fit_templates(backgrounds,seed=20261029,n_clusters=48):
    rng=np.random.default_rng(seed)
    if len(backgrounds)>24000:backgrounds=backgrounds[rng.choice(len(backgrounds),24000,replace=False)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        spectra=np.nanmedian(backgrounds,axis=1)
    offset=np.nanmedian(spectra,axis=1,keepdims=True)
    shape=np.nan_to_num(spectra-offset,nan=0.)
    km=MiniBatchKMeans(n_clusters=n_clusters,batch_size=2048,n_init=3,random_state=seed,max_iter=150)
    km.fit(shape)
    return km.cluster_centers_.astype(np.float32)

def feature_batch(db,templates=None):
    db=np.asarray(db,dtype=np.float32)
    mask=np.isfinite(db);n=len(db);cols=[];names=[]
    def add(x,name):
        x=np.asarray(x).reshape(n,-1)
        cols.append(x);names.extend([name] if x.shape[1]==1 else [f'{name}_{j:02d}' for j in range(x.shape[1])])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        mean=np.nanmean(db,1);std=np.nanstd(db,1)
        q=np.nanpercentile(db,[10,25,50,75,90],axis=1)
        fullmean=np.nanmean(db,axis=(1,2));fullstd=np.nanstd(db,axis=(1,2))
        e,offset=energy(db)
        count=mask.sum((1,2));total=e.sum((1,2))+1e-25
        totals_db=10*np.log10(total/np.maximum(count,1))+offset[:,0,0]
        band_count=mask.sum(1)
        band_e=e.sum(1)/np.maximum(band_count,1)
        band_db=10*np.log10(np.maximum(band_e,1e-25))+offset[:,0,:]
        add(fullmean,'global_mean');add(fullstd,'global_std');add(totals_db,'received_energy_db')
        add(mask.mean((1,2)),'coverage');add(mask.mean(1),'band_coverage');add(mask.mean(2),'minute_coverage')
        add(mean,'band_mean');add(std,'band_std');add(q[2],'band_median')
        add(q[4]-q[0],'band_range90_10');add(q[3]-q[1],'band_iqr')
        add(mean-fullmean[:,None],'relative_spectrum');add(band_db,'band_energy_db')
        add(band_db-mean,'energy_arithmetic_excess');add(np.diff(mean,axis=1),'spectral_difference')
        add(np.diff(mean,n=2,axis=1),'spectral_curvature')
        band_frac=e.sum(1)/total[:,None]
        add(band_frac,'band_energy_fraction')
        delta=np.diff(db,axis=1)
        add(np.nanmean(np.abs(delta),1),'time_mean_abs_difference')
        add(np.nanstd(delta,1),'time_std_difference')
        add(np.nanmean(np.abs(np.diff(db,n=2,axis=1)),1),'time_second_difference')
        # Temporal summaries over broad frequency ranges.
        for lo,hi in [(0,25),(0,8),(8,17),(17,25),(0,4),(4,12),(12,21)]:
            name=f'group{lo}_{hi}'
            eg=e[:,:,lo:hi];mg=mask[:,:,lo:hi]
            trajectory=eg.sum(2)/np.maximum(mg.sum(2),1)
            trajectory=np.where(mg.any(2),trajectory,np.nan)
            level=10*np.log10(np.maximum(trajectory,1e-25))+offset[:,0,:]
            center=np.nanmean(level,1,keepdims=True)
            add(np.nanmean(db[:,:,lo:hi],(1,2)),name+'_db')
            add(eg.sum((1,2))/total,name+'_share')
            add(np.nanstd(level,1),name+'_std')
            add(np.nanmax(level,1)-np.nanmin(level,1),name+'_range')
            add(np.nanmean(np.abs(np.diff(level,axis=1)),1),name+'_roughness')
            rel=np.nan_to_num(level-center,nan=0.)
            add(np.sum(rel[:,:-1]*rel[:,1:],1)/(np.sum(rel*rel,1)+1e-6),name+'_lag1')
            add(np.sum(rel[:,:-2]*rel[:,2:],1)/(np.sum(rel*rel,1)+1e-6),name+'_lag2')
            add(np.nanmean(level[:,:5],1)-np.nanmean(level[:,-5:],1),name+'_edge_difference')
            add(np.nanmean(level[:,5:10],1)-.5*(np.nanmean(level[:,:5],1)+np.nanmean(level[:,-5:],1)),name+'_center_bulge')
            baseline=np.nanpercentile(trajectory,20,axis=1)
            add(np.nansum(np.maximum(trajectory-baseline[:,None],0),1)/(np.nansum(trajectory,1)+1e-25),name+'_temporal_excess')
            add(np.nanmax(trajectory,1)/(np.nansum(trajectory,1)+1e-25),name+'_peak_fraction')
            if lo in [0,8,17] and hi in [25,17]:add(rel,name+'_trajectory')
        # Coherent changes across bands, without relying on sub-dB export rounding.
        centered=np.nan_to_num(db-mean[:,None,:],nan=0.)
        singular=np.linalg.svd(centered,compute_uv=False)
        ev=singular**2;ev/=ev.sum(1,keepdims=True)+1e-8
        add(ev[:,:5],'temporal_singular_energy')
        add(-np.sum(band_frac*np.log(np.maximum(band_frac,1e-25)),1),'spectral_entropy')
        add(np.sort(band_frac,axis=1)[:,-5:],'largest_band_shares')
        if templates is not None:
            # Broad clustered spectral shapes, never a nearest-recording lookup.
            diff=mean[:,None,:]-templates[None,:,:]
            shift=np.nanmedian(diff,axis=2)
            resid=diff-shift[:,:,None]
            errors=np.nanmean(np.minimum(resid**2,100),axis=2)
            best=np.argsort(errors,axis=1)[:,:3]
            for k in range(3):
                index=best[:,k];template=templates[index]
                difference=band_db-template
                lower_shift=np.nanpercentile(np.where(band_count>0,difference,np.nan),20,axis=1)
                base=np.power(10.,np.clip((template+lower_shift[:,None]-offset[:,0,:])/10,-25,10))
                excess=np.maximum(band_e-base,0)*band_count
                add(excess.sum(1)/total,f'template{k}_excess')
                add(errors[np.arange(n),index],f'template{k}_error')
            add(resid[np.arange(n),best[:,0]],'template_residual')
    features=np.concatenate(cols,axis=1).astype(np.float32)
    features[~np.isfinite(features)]=np.nan
    return features,names

def features(db,templates=None,batch_size=1024):
    result=[];names=None
    for i in range(0,len(db),batch_size):
        f,names=feature_batch(db[i:i+batch_size],templates)
        result.append(f)
    return np.concatenate(result),names
