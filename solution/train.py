"""Manual model comparisons on train-only simulations; never opens Test.csv."""
from pathlib import Path
import argparse,gc,json,os,platform,resource,time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
from acoustic_model import fit_templates,features
from simulation import Simulator,BASE

SCENARIOS=['standard','weak','strong','channel_shift','missing','flat']
SCENARIO_WEIGHTS={'standard':.4,'weak':.25,'strong':.1,'channel_shift':.1,'missing':.1,'flat':.05}

def metrics(y,p,meta):
    result={'rmse':float(np.sqrt(np.mean((y-p)**2))),'mae':float(np.mean(abs(y-p))),'bias':float(np.mean(p-y))}
    if meta.is_added.nunique()>1:result['diagnostic_auc']=float(roc_auc_score(meta.is_added,p))
    for name,ix in [('no_addition',y==0),('weak', (y>0)&(y<=.05)),('medium',(y>.05)&(y<=.2)),('strong',y>.2)]:
        if ix.any():result[name]={'n':int(ix.sum()),'rmse':float(np.sqrt(np.mean((y[ix]-p[ix])**2))),'prediction_mean':float(p[ix].mean()),'target_mean':float(y[ix].mean())}
    return result

def save_json(path,data):path.write_text(json.dumps(data,indent=2))

def generate_training(sim,n,seed,templates,out):
    parts=[];ys=[];metas=[]
    for i in range(0,n,8000):
        x,y,meta=sim.generate(min(8000,n-i),seed+i)
        f,names=features(x,templates)
        parts.append(f);ys.append(y);metas.append(meta)
        print(f'Features {i+len(y)}/{n}, {f.shape[1]} columns',flush=True)
    x=np.concatenate(parts);y=np.concatenate(ys);meta=pd.concat(metas,ignore_index=True)
    meta.to_csv(out/'training_examples.csv.gz',index=False)
    save_json(out/'feature_names.json',names)
    return x,y,meta,names

def build_validation(sim,n,templates,out):
    result={}
    for j,scenario in enumerate(SCENARIOS):
        x,y,meta=sim.generate(n,800000+j*10000,scenario=scenario,augment_background=False)
        f,names=features(x,templates)
        result[scenario]=(f,y,meta)
        meta.to_csv(out/f'validation_{scenario}.csv.gz',index=False)
    return result

def evaluate(models,validation,out):
    scores={};predictions={}
    for name,model in models.items():
        scores[name]={};predictions[name]={}
        for scenario,(x,y,meta) in validation.items():
            p=np.clip(model.predict(x),0,1) if hasattr(model,'predict') else np.full(len(y),model)
            predictions[name][scenario]=p
            scores[name][scenario]=metrics(y,p,meta)
        scores[name]['weighted_rmse']=sum(SCENARIO_WEIGHTS[s]*scores[name][s]['rmse'] for s in SCENARIOS)
        print(name,'weighted RMSE',scores[name]['weighted_rmse'],flush=True)
    # A small, explicitly defined convex blend; no automated hyperparameter search.
    if 'lightgbm' in predictions and 'catboost' in predictions:
        scores['blend']={};predictions['blend']={}
        for scenario,(x,y,meta) in validation.items():
            p=.5*predictions['lightgbm'][scenario]+.5*predictions['catboost'][scenario]
            predictions['blend'][scenario]=p;scores['blend'][scenario]=metrics(y,p,meta)
        scores['blend']['weighted_rmse']=sum(SCENARIO_WEIGHTS[s]*scores['blend'][s]['rmse'] for s in SCENARIOS)
        print('blend','weighted RMSE',scores['blend']['weighted_rmse'],flush=True)
    save_json(out/'validation_metrics.json',scores)
    for scenario,(x,y,meta) in validation.items():
        frame=meta.copy()
        for name in predictions:frame[name]=predictions[name][scenario]
        frame.to_csv(out/f'validation_predictions_{scenario}.csv.gz',index=False)
    return scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='development');ap.add_argument('--n-train',type=int,default=80000);ap.add_argument('--n-val',type=int,default=2500);ap.add_argument('--iterations',type=int,default=1600);ap.add_argument('--conservative-iterations',type=int);ap.add_argument('--catboost-iterations',type=int);ap.add_argument('--prototype',action='store_true');ap.add_argument('--final',action='store_true');args=ap.parse_args()
    started=time.time();out=BASE/'runs'/args.name;out.mkdir(parents=True,exist_ok=True)
    train_splits=['train'] if not args.final else ['train','val_time','val_site','audit_time','audit_site']
    ref_splits=['train'] if not args.final else ['train','val','audit']
    sim=Simulator(train_splits,ref_splits)
    if not args.prototype:assert (BASE/'cache/reference_manifest.csv').exists(),'Wait for complete fixed source selection.'
    save_json(out/'config.json',{'args':vars(args),'training_background_splits':train_splits,'training_reference_splits':ref_splits,'seed':20261029,'source_recordings':len(sim.refs),'feature_version':1,'test_inputs_read':False,'scenario_weights':SCENARIO_WEIGHTS})
    sim.refs.to_csv(out/'training_reference_manifest.csv',index=False)
    # Templates only use the same permitted training partition as the supervised model.
    rng=np.random.default_rng(20261029)
    template_idx=rng.choice(sim.allowed,min(24000,len(sim.allowed)),replace=False,p=sim.weights)
    templates=fit_templates(np.array(sim.backgrounds[template_idx]))
    np.save(out/'templates.npy',templates)
    print('Fitted training-only background templates',flush=True)
    x,y,meta,names=generate_training(sim,args.n_train,20261029,templates,out)
    validation={} if args.final else build_validation(Simulator(['val_time','val_site'],['val']),args.n_val,templates,out)
    baseline_cols=[names.index(x) for x in ['global_mean','global_std','received_energy_db','coverage']]
    baseline=lgb.LGBMRegressor(n_estimators=250,num_leaves=12,learning_rate=.04,min_child_samples=100,verbosity=-1,n_jobs=5,random_state=101,deterministic=True,force_col_wise=True)
    baseline.fit(x[:,baseline_cols],y)
    joblib.dump({'model':baseline,'columns':baseline_cols},out/'energy_baseline.joblib')
    model=lgb.LGBMRegressor(n_estimators=args.iterations,num_leaves=48,learning_rate=.035,max_depth=-1,min_child_samples=70,colsample_bytree=.85,subsample=.85,subsample_freq=1,reg_lambda=8,reg_alpha=.1,max_bin=127,n_jobs=6,verbosity=-1,random_state=20261029,deterministic=True,force_col_wise=True)
    if validation:
        vx,vy,_=validation['standard']
        model.fit(x,y,eval_set=[(vx,vy)],callbacks=[lgb.early_stopping(150),lgb.log_evaluation(100)],feature_name=names)
    else:model.fit(x,y,feature_name=names,callbacks=[lgb.log_evaluation(200)])
    joblib.dump(model,out/'lightgbm.joblib')
    print('LightGBM fit complete',model.best_iteration_,flush=True)
    # A separately documented low-addition prior, for sensitivity analysis only.
    conservative=lgb.LGBMRegressor(**model.get_params())
    conservative.set_params(n_estimators=args.conservative_iterations or args.iterations)
    weights=np.exp(-4*y)
    if validation:
        conservative.fit(x,y,sample_weight=weights,eval_set=[(vx,vy)],eval_sample_weight=[np.exp(-4*vy)],callbacks=[lgb.early_stopping(150),lgb.log_evaluation(200)],feature_name=names)
    else:conservative.fit(x,y,sample_weight=weights,feature_name=names)
    joblib.dump(conservative,out/'conservative.joblib')
    cat=CatBoostRegressor(iterations=args.catboost_iterations or args.iterations,depth=7,learning_rate=.045,loss_function='RMSE',l2_leaf_reg=10,random_seed=731,thread_count=6,allow_writing_files=False,verbose=200)
    # CatBoost uses a finite missing sentinel; real levels/features never approach this value.
    xc=np.nan_to_num(x,nan=-9999.,posinf=-9999.,neginf=-9999.)
    if validation:
        cat.fit(xc,y,eval_set=(np.nan_to_num(vx,nan=-9999.),vy),early_stopping_rounds=150,use_best_model=True)
    else:cat.fit(xc,y)
    cat.save_model(str(out/'catboost.cbm'))
    classifier=lgb.LGBMClassifier(n_estimators=min(args.iterations,900),num_leaves=31,learning_rate=.04,min_child_samples=100,colsample_bytree=.8,reg_lambda=10,max_bin=127,n_jobs=6,verbosity=-1,random_state=371,deterministic=True,force_col_wise=True)
    if validation:
        classifier.fit(x,meta.is_added,eval_set=[(vx,validation['standard'][2].is_added)],eval_metric='auc',callbacks=[lgb.early_stopping(100),lgb.log_evaluation(200)])
    else:classifier.fit(x,meta.is_added)
    joblib.dump(classifier,out/'evidence.joblib')
    save_json(out/'training_summary.json',{'wall_seconds':time.time()-started,'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'cpu_count':os.cpu_count(),'platform':platform.platform(),'lightgbm_iterations':model.n_estimators_,'conservative_iterations':conservative.n_estimators_,'evidence_iterations':classifier.n_estimators_,'catboost_iterations':cat.tree_count_,'n_train':len(y),'reference_clips':len(sim.refs),'target_mean':float(y.mean()),'conservative_target_mean':float(np.average(y,weights=weights)),'addition_rate':float(meta.is_added.mean()),'target_quantiles':np.quantile(y,[0,.1,.25,.5,.75,.9,.99,1]).tolist()})
    if validation:
        class Wrapped:
            def __init__(self,model,columns=None,fill=False):self.model=model;self.columns=columns;self.fill=fill
            def predict(self,x):
                if self.columns is not None:x=x[:,self.columns]
                if self.fill:x=np.nan_to_num(x,nan=-9999.)
                return self.model.predict(x)
        scores=evaluate({'constant':float(y.mean()),'received_energy':Wrapped(baseline,baseline_cols),'lightgbm':model,'conservative':conservative,'catboost':Wrapped(cat,fill=True)},validation,out)
        table=pd.DataFrame({name:{s:sc[s]['rmse'] for s in SCENARIOS} for name,sc in scores.items()})
        print(table.to_string(),flush=True)
        table.to_csv(out/'validation_summary.csv')
        evidence_scores={}
        for scenario,(vx,vy,vm) in validation.items():
            ev=classifier.predict_proba(vx)[:,1]
            evidence_scores[scenario]=float(roc_auc_score(vm.is_added,ev))
            pd.DataFrame({'evidence':ev}).to_csv(out/f'validation_evidence_{scenario}.csv.gz',index=False)
        save_json(out/'evidence_metrics.json',evidence_scores)
    pd.DataFrame({'feature':names,'gain':model.booster_.feature_importance('gain')}).sort_values('gain',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    print('Finished',out,flush=True)

if __name__=='__main__':main()
