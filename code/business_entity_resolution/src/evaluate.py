"""Exact Source-1 entity macro F0.5, including empty-truth singletons."""
import numpy as np

def metric(scores,labels,groups,truth_counts,n_entities,threshold):
    take=np.asarray(scores)>=threshold
    groups=np.asarray(groups,dtype=np.int32);labels=np.asarray(labels,dtype=np.int8)
    pred=np.bincount(groups,weights=take,minlength=n_entities)
    tp=np.bincount(groups,weights=take& (labels==1),minlength=n_entities)
    tc=np.asarray(truth_counts,dtype=np.float64)
    fp=pred-tp;fn=tc-tp
    den=1.25*tp+fp+0.25*fn
    per=np.divide(1.25*tp,den,out=np.zeros_like(tp),where=den>0)
    per[(tc==0)&(pred==0)]=1.0
    TP=int(tp.sum());FP=int(fp.sum());FN=int(fn.sum())
    sing=tc==0
    return dict(macro_f0_5=float(per.mean()),precision=float(TP/(TP+FP)) if TP+FP else 1.0,recall=float(TP/(TP+FN)) if TP+FN else 1.0,singleton_accuracy=float(((pred==0)&sing).sum()/sing.sum()) if sing.any() else 0.0,singletons=int(sing.sum()),predicted_links=int(pred.sum()),false_positives=FP,false_negatives=FN,threshold=float(threshold))

def search_threshold(scores,labels,groups,truth_counts,n_entities):
    broad=[.1,.2,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.82,.84,.86,.88,.9,.91,.92,.93,.94,.95,.96,.97,.98,.99,.995,.999]
    rows=[metric(scores,labels,groups,truth_counts,n_entities,t) for t in broad]
    best=max(rows,key=lambda r:r['macro_f0_5'])
    lo=max(0,best['threshold']-.025);hi=min(1,best['threshold']+.025)
    rows += [metric(scores,labels,groups,truth_counts,n_entities,t) for t in np.linspace(lo,hi,51)]
    return max(rows,key=lambda r:r['macro_f0_5']),rows
