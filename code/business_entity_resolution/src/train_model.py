import csv,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from blocking import connect,retrieve
from features import FEATURE_NAMES,pair_features,rank_candidates
from evaluate import metric,search_threshold

def prepare(data_root,index_path,cache_path,k=50,top=50):
    data_root=Path(data_root);cache_path=Path(cache_path)
    if cache_path.exists():
        z=np.load(cache_path,allow_pickle=False)
        return {x:z[x] for x in z.files}
    truth={}
    for chunk in pd.read_csv(data_root/'train'/'train_ground_truth.tsv',sep='\t',dtype=str,keep_default_na=False,chunksize=100000):
        for r in chunk.itertuples(index=False):
            num=int(r.source1_entity_id[3:])
            if num%200==0 or num%1000==1:truth[r.source1_entity_id]=set(r.matched_entity_ids.split(','))-{''}
    X=[];y=[];groups=[];splits=[];truth_counts=[];candidate_hits=[]
    con=connect(str(index_path)); t=time.time()
    for chunk in pd.read_csv(data_root/'train'/'train_source1.tsv',sep='\t',dtype=str,keep_default_na=False,chunksize=100000):
        for r in chunk.itertuples(index=False):
            if r.entity_id not in truth:continue
            gi=len(truth_counts);tc=truth[r.entity_id]; truth_counts.append(len(tc))
            split=int(int(r.entity_id[3:])%1000==1);splits.append(split)
            candidates=rank_candidates(r.business_name,r.business_address,retrieve(con,r.business_name,r.business_address,r.country,k,k),top)
            candidate_hits.append(sum(row[0][0] in tc for row in candidates))
            for (mid,mn,ma,mc),nf,af in candidates:
                X.append(pair_features(r.business_name,r.business_address,r.country,mn,ma,mc,nf,af))
                y.append(int(mid in tc));groups.append(gi)
        if len(truth_counts)%1000<100:print('sample entities',len(truth_counts),'pairs',len(y),'seconds',round(time.time()-t),flush=True)
    con.close()
    z=dict(X=np.vstack(X),y=np.asarray(y,dtype=np.uint8),groups=np.asarray(groups,dtype=np.int32),splits=np.asarray(splits,dtype=np.uint8),truth_counts=np.asarray(truth_counts,dtype=np.int16),candidate_hits=np.asarray(candidate_hits,dtype=np.int16))
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache_path,**z)
    return z

def subset(z,split):
    entity_idx=np.flatnonzero(z['splits']==split)
    rev=np.full(len(z['splits']),-1,dtype=np.int32);rev[entity_idx]=np.arange(len(entity_idx))
    mask=np.isin(z['groups'],entity_idx)
    return z['X'][mask],z['y'][mask],rev[z['groups'][mask]],z['truth_counts'][entity_idx],z['candidate_hits'][entity_idx]

def train(data_root,index_path,cache_path,model_dir,report_dir,k=50,top=50):
    model_dir=Path(model_dir);report_dir=Path(report_dir);model_dir.mkdir(parents=True,exist_ok=True);report_dir.mkdir(parents=True,exist_ok=True)
    z=prepare(data_root,index_path,cache_path,k,top)
    Xtr,ytr,gtr,ttr,htr=subset(z,0);Xv,yv,gv,tv,hv=subset(z,1)
    print('train',Xtr.shape,'validation',Xv.shape,'candidate recall',sum(hv)/sum(tv),flush=True)
    configs=[
        ('weighted_similarity',None),
        ('logistic',make_pipeline(StandardScaler(),LogisticRegression(max_iter=300,class_weight='balanced',random_state=42))),
        ('lgbm_small',LGBMClassifier(n_estimators=220,learning_rate=.06,num_leaves=31,max_depth=-1,min_child_samples=70,colsample_bytree=.85,subsample=.85,reg_lambda=2,verbosity=-1,random_state=42,n_jobs=4)),
        ('lgbm_regularized',LGBMClassifier(n_estimators=350,learning_rate=.04,num_leaves=15,min_child_samples=150,colsample_bytree=.8,subsample=.8,reg_lambda=5,verbosity=-1,random_state=42,n_jobs=4)),
        ('lgbm_deeper',LGBMClassifier(n_estimators=350,learning_rate=.04,num_leaves=63,min_child_samples=100,colsample_bytree=.85,subsample=.85,reg_lambda=2,verbosity=-1,random_state=42,n_jobs=4)),
    ]
    leaderboard=[];curves=[];best_score=-1;chosen=None
    for name,model in configs:
        start=time.time()
        if model is None:
            scores=.58*Xv[:,0]+.42*Xv[:,14]
        else:
            model.fit(Xtr,ytr)
            scores=model.predict_proba(Xv)[:,1]
        best,curve=search_threshold(scores,yv,gv,tv,len(tv))
        for x in curve:curves.append(dict(experiment=name,**x))
        row=dict(experiment=name,candidate_strategy='FTS5 rare-token name+address',top_k_name=k,top_k_address=k,final_candidates_per_entity=top,candidate_recall=float(sum(hv)/sum(tv)),feature_set='full' if model else 'weighted name/address',model=name,model_params=str(model.get_params()) if model else '{}',runtime=round(time.time()-start,1),**best)
        leaderboard.append(row);print(name,best,flush=True)
        if best['macro_f0_5']>best_score:
            best_score=best['macro_f0_5'];chosen=(name,model,best)
    pd.DataFrame(leaderboard).sort_values('macro_f0_5',ascending=False).to_csv(report_dir/'experiment_results.csv',index=False)
    pd.DataFrame(curves).to_csv(report_dir/'threshold_search.csv',index=False)
    name,model,best=chosen
    if model is not None:
        model.fit(z['X'],z['y'])
        joblib.dump(model,model_dir/'best_model.joblib')
    metadata=dict(experiment=name,threshold=best['threshold'],feature_names=FEATURE_NAMES,candidate_k=k,final_candidates=top,validation=best,candidate_recall=float(sum(hv)/sum(tv)),train_entities=len(ttr),validation_entities=len(tv),seed=42,model_params=model.get_params() if name.startswith('lgbm') else {})
    (model_dir/'selection.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata
