"""Memory-bounded validation of both submission files against test Source 1."""
from pathlib import Path
import pandas as pd

def validate_streaming(data_root,output_dir):
    data_root=Path(data_root);output_dir=Path(output_dir)
    s1=pd.read_csv(data_root/'test'/'test_source1.tsv',sep='\t',dtype=str,keep_default_na=False,chunksize=10000)
    can=pd.read_csv(output_dir/'candidate_pairs.tsv',sep='\t',dtype=str,keep_default_na=False,chunksize=10000)
    mat=pd.read_csv(output_dir/'matching_results.tsv',sep='\t',dtype=str,keep_default_na=False,chunksize=10000)
    n=links=0
    for a,b,c in zip(s1,can,mat,strict=True):
        if list(b.columns)!=['source1_entity_id','candidate_entity_ids']:raise ValueError('Bad candidate columns')
        if list(c.columns)!=['source1_entity_id','matched_entity_ids']:raise ValueError('Bad matching columns')
        if len(a)!=len(b) or len(a)!=len(c):raise ValueError('Row count mismatch')
        for sid,bid,mid,cids,mids in zip(a.entity_id,b.source1_entity_id,c.source1_entity_id,b.candidate_entity_ids,c.matched_entity_ids,strict=True):
            if sid!=bid or sid!=mid:raise ValueError(f'Source 1 ID mismatch near row {n+1}')
            candidate=cids.split(',') if cids else []
            matched=mids.split(',') if mids else []
            if len(candidate)!=len(set(candidate)) or len(matched)!=len(set(matched)):raise ValueError(f'Duplicate ID near row {n+1}')
            if not set(matched)<=set(candidate):raise ValueError(f'Match absent from candidates near row {n+1}')
            if any(not x.startswith(('S2-','S3-')) for x in candidate):raise ValueError(f'Bad candidate prefix near row {n+1}')
            n+=1;links+=len(matched)
    print(f'STREAMING PASS: {n} Source 1 rows, {links} matched links; complete coverage and match subset candidate verified.',flush=True)
    return n,links
