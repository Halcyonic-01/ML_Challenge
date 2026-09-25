"""Ablations, negative sampling, and an untouched second entity holdout."""
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from train_model import subset
from evaluate import search_threshold,metric
from blocking import connect,retrieve
from features import pair_features,rank_candidates

BASE=dict(n_estimators=220,learning_rate=.06,num_leaves=31,min_child_samples=70,colsample_bytree=.85,subsample=.85,reg_lambda=2,verbosity=-1,random_state=42,n_jobs=4)
DEEP=dict(BASE,n_estimators=350,learning_rate=.04,num_leaves=63,min_child_samples=100)

def blocking_study(data_root,index_path,report_dir):
    data_root=Path(data_root);truth={}
    for chunk in pd.read_csv(data_root/'train'/'train_ground_truth.tsv',sep='\t',dtype=str,keep_default_na=False,chunksize=100000):
        for r in chunk.itertuples(index=False):
            if int(r.source1_entity_id[3:])%1000==0:truth[r.source1_entity_id]=set(r.matched_entity_ids.split(','))-{''}
    con=connect(str(index_path));target_count=con.execute('SELECT COUNT(*) FROM targets').fetchone()[0];records=[]
    for chunk in pd.read_csv(data_root/'train'/'train_source1.tsv',sep='\t',dtype=str,keep_default_na=False,chunksize=100000):
        records.extend((r.entity_id,r.business_name,r.business_address,r.country) for r in chunk.itertuples(index=False) if r.entity_id in truth)
    rows=[]
    for k in (20,50):
        counts={n:0 for n in ['links','name_found','address_found','raw_found','raw_candidates']}
        raw_sizes=[];cap_sizes={cap:[] for cap in (10,20,30,40,50,75,100)}
        for cap in (10,20,30,40,50,75,100):counts[f'found_{cap}']=0;counts[f'candidates_{cap}']=0
        start=time.time()
        for sid,name,address,country in records:
            matches=truth[sid];found=retrieve(con,name,address,country,k,k)
            ids={x[0][0] for x in found};nids={x[0][0] for x in found if x[1]};aids={x[0][0] for x in found if x[2]}
            raw_sizes.append(len(ids))
            counts['links']+=len(matches);counts['raw_found']+=len(ids&matches);counts['raw_candidates']+=len(ids)
            counts['name_found']+=len(nids&matches);counts['address_found']+=len(aids&matches)
            ranked=rank_candidates(name,address,found,100)
            for cap in (10,20,30,40,50,75,100):
                selected={x[0][0] for x in ranked[:cap]};counts[f'found_{cap}']+=len(selected&matches);counts[f'candidates_{cap}']+=len(selected)
                cap_sizes[cap].append(len(selected))
        for cap in (10,20,30,40,50,75,100):
            rows.append(dict(top_k=k,final_cap=cap,entities=len(records),raw_recall=counts['raw_found']/counts['links'],candidate_recall=counts[f'found_{cap}']/counts['links'],name_only_recall=counts['name_found']/counts['links'],address_only_recall=counts['address_found']/counts['links'],avg_raw_candidates=counts['raw_candidates']/len(records),median_raw_candidates=float(np.median(raw_sizes)),max_raw_candidates=max(raw_sizes),avg_final_candidates=counts[f'candidates_{cap}']/len(records),median_final_candidates=float(np.median(cap_sizes[cap])),max_final_candidates=max(cap_sizes[cap]),candidate_reduction_ratio=1-counts[f'candidates_{cap}']/(len(records)*target_count),runtime_seconds=round(time.time()-start,1)))
    con.close();pd.DataFrame(rows).to_csv(Path(report_dir)/'blocking_results.csv',index=False)

def run_ablations(cache_path,report_dir):
    z=np.load(cache_path);Xtr,ytr,_,_,_=subset(z,0);Xv,yv,gv,tv,hv=subset(z,1)
    pos=np.flatnonzero(ytr==1);neg=np.flatnonzero(ytr==0);rng=np.random.default_rng(42)
    hardness=np.maximum(Xtr[neg,0],Xtr[neg,14]);rows=[];curves=[]
    configs=[('without_address',BASE,list(range(14))+[28,30],None),('without_name',BASE,list(range(14,28))+[29,30],None),('without_numeric_address',BASE,[i for i in range(36) if i not in (22,23,24,25)],None),('without_retrieval_flags',BASE,[i for i in range(36) if i not in (28,29)],None),('hard_negatives_5to1',BASE,list(range(36)),'hard5'),('mixed_negatives_10to1',BASE,list(range(36)),'mix10')]
    for name,params,cols,sample in configs:
        start=time.time();ix=np.arange(len(ytr))
        if sample=='hard5':
            h=min(len(neg),len(pos)*5);ix=np.concatenate([pos,neg[np.argpartition(hardness,-h)[-h:]]])
        elif sample=='mix10':
            h=min(len(neg),len(pos)*8);hard=neg[np.argpartition(hardness,-h)[-h:]]
            rest=np.setdiff1d(neg,hard);ix=np.concatenate([pos,hard,rng.choice(rest,min(len(rest),len(pos)*2),replace=False)])
        model=LGBMClassifier(**params);model.fit(Xtr[ix][:,cols],ytr[ix]);scores=model.predict_proba(Xv[:,cols])[:,1]
        best,curve=search_threshold(scores,yv,gv,tv,len(tv))
        rows.append(dict(experiment=name,candidate_strategy='FTS5 rare-token name+address',top_k_name=50,top_k_address=50,final_candidates_per_entity=75,candidate_recall=float(hv.sum()/tv.sum()),feature_set=name,model='LightGBM',model_params=json.dumps(params),runtime=round(time.time()-start,1),**best))
        curves.extend(dict(experiment=name,**x) for x in curve)
    report_dir=Path(report_dir)
    main=pd.read_csv(report_dir/'experiment_results.csv');pd.concat([main,pd.DataFrame(rows)],ignore_index=True).sort_values('macro_f0_5',ascending=False).to_csv(report_dir/'experiment_results.csv',index=False)
    main=pd.read_csv(report_dir/'threshold_search.csv');pd.concat([main,pd.DataFrame(curves)],ignore_index=True).to_csv(report_dir/'threshold_search.csv',index=False)

def second_holdout(data_root,index_path,cache_path,model_dir,report_dir):
    data_root=Path(data_root);model_dir=Path(model_dir);report_dir=Path(report_dir)
    cfg=json.loads((model_dir/'selection.json').read_text(encoding='utf-8'))
    truth={}
    for chunk in pd.read_csv(data_root/'train'/'train_ground_truth.tsv',sep='\t',dtype=str,keep_default_na=False,chunksize=100000):
        for r in chunk.itertuples(index=False):
            if int(r.source1_entity_id[3:])%1000==2:truth[r.source1_entity_id]=set(r.matched_entity_ids.split(','))-{''}
    con=connect(str(index_path));X=[];y=[];groups=[];tc=[];hits=[]
    for chunk in pd.read_csv(data_root/'train'/'train_source1.tsv',sep='\t',dtype=str,keep_default_na=False,chunksize=100000):
        for r in chunk.itertuples(index=False):
            if r.entity_id not in truth:continue
            gi=len(tc);ids=truth[r.entity_id];tc.append(len(ids))
            cands=rank_candidates(r.business_name,r.business_address,retrieve(con,r.business_name,r.business_address,r.country,cfg['candidate_k'],cfg['candidate_k']),cfg['final_candidates'])
            hits.append(sum(item[0][0] in ids for item in cands))
            for (mid,mn,ma,mc),nf,af in cands:
                X.append(pair_features(r.business_name,r.business_address,r.country,mn,ma,mc,nf,af));y.append(int(mid in ids));groups.append(gi)
    con.close();X=np.vstack(X);y=np.asarray(y);groups=np.asarray(groups);tc=np.asarray(tc)
    z=np.load(cache_path);Xtr,ytr,_,_,_=subset(z,0)
    if not cfg['experiment'].startswith('lgbm'):raise ValueError('Second holdout expects a LightGBM model')
    params=cfg.get('model_params') or DEEP
    model=LGBMClassifier(**params);model.fit(Xtr,ytr);scores=model.predict_proba(X)[:,1]
    result=dict(candidate_recall=float(sum(hits)/sum(tc)),entities=len(tc),**metric(scores,y,groups,tc,len(tc),cfg['threshold']))
    (report_dir/'second_holdout.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    cfg['second_holdout']=result;(model_dir/'selection.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print('second holdout',result,flush=True)
    return result

def singleton_rules(cache_path,report_dir):
    z=np.load(cache_path);Xtr,ytr,_,_,_=subset(z,0);Xv,yv,gv,tv,_=subset(z,1)
    model=LGBMClassifier(**DEEP);model.fit(Xtr,ytr);scores=model.predict_proba(Xv)[:,1]
    base=.537;rows=[dict(rule='global',parameter=base,**metric(scores,yv,gv,tv,len(tv),base))]
    pred_count=np.bincount(gv,weights=scores>=base,minlength=len(tv));top=np.zeros(len(tv));second=np.zeros(len(tv))
    for g,score in zip(gv,scores):
        if score>top[g]:second[g]=top[g];top[g]=score
        elif score>second[g]:second[g]=score
    for t in (.6,.65,.7,.75,.8,.85,.9,.95):
        adjusted=scores.copy();adjusted[(pred_count[gv]==1)&(top[gv]<t)]=0
        rows.append(dict(rule='single_link_min_top',parameter=t,**metric(adjusted,yv,gv,tv,len(tv),base)))
    for margin in (.01,.02,.03,.05,.08,.1,.15):
        adjusted=scores.copy();adjusted[(pred_count[gv]==1)&((top-second)[gv]<margin)]=0
        rows.append(dict(rule='single_link_margin',parameter=margin,**metric(adjusted,yv,gv,tv,len(tv),base)))
    pd.DataFrame(rows).sort_values('macro_f0_5',ascending=False).to_csv(Path(report_dir)/'singleton_rules.csv',index=False)
