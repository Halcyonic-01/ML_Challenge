"""Country-agnostic, ID-free pair features."""
import re
from functools import lru_cache
import numpy as np
from rapidfuzz import fuzz, distance
from blocking import norm, tokens

LEGAL={'pvt':'private','ltd':'limited','corp':'corporation','inc':'incorporated','co':'company','llc':'limited liability company'}
FEATURE_NAMES=['name_ratio','name_token_sort','name_token_set','name_partial','name_jaro','name_legal_ratio','name_exact','name_sorted_exact','name_jaccard','name_common','name_first_equal','name_last_equal','name_len_ratio','name_digits_jaccard','address_ratio','address_token_sort','address_token_set','address_partial','address_exact','address_jaccard','address_common','address_len_ratio','address_num_jaccard','address_num_exact','address_num_conflict','address_postal_equal','address_missing','candidate_address_missing','retrieved_name','retrieved_address','country_equal','min_similarity','max_similarity','similarity_product','name_strong_address_weak','address_strong_name_weak']

@lru_cache(maxsize=20000)
def parts(s):
    v=norm(s); seq=v.split();ts=set(seq)
    digits={x for x in ts if any(c.isdigit() for c in x)}
    post={x for x in digits if len(x)>=5}
    return v,seq,ts,digits,post

def jac(a,b):
    if not a or not b:return 0.0
    return len(a&b)/len(a|b)

@lru_cache(maxsize=20000)
def legal(s):
    return ' '.join(LEGAL.get(x,x) for x in tokens(s))

def pair_features(name1,addr1,country1,name2,addr2,country2,from_name=0,from_address=0):
    n1,sn1,tn1,dn1,_=parts(name1);n2,sn2,tn2,dn2,_=parts(name2)
    a1,sa1,ta1,da1,p1=parts(addr1);a2,sa2,ta2,da2,p2=parts(addr2)
    nr=fuzz.ratio(n1,n2)/100; ar=fuzz.ratio(a1,a2)/100 if a1 and a2 else 0
    ns=fuzz.token_set_ratio(n1,n2)/100; ass=fuzz.token_set_ratio(a1,a2)/100 if a1 and a2 else 0
    return np.asarray([
        nr,fuzz.token_sort_ratio(n1,n2)/100,ns,fuzz.partial_ratio(n1,n2)/100,distance.JaroWinkler.normalized_similarity(n1,n2),fuzz.ratio(legal(name1),legal(name2))/100,
        int(n1==n2),int(sorted(tn1)==sorted(tn2)),jac(tn1,tn2),len(tn1&tn2),int(bool(sn1 and sn2) and sn1[0]==sn2[0]),int(bool(sn1 and sn2) and sn1[-1]==sn2[-1]),min(len(n1),len(n2))/max(1,len(n1),len(n2)),jac(dn1,dn2),
        ar,fuzz.token_sort_ratio(a1,a2)/100 if a1 and a2 else 0,ass,fuzz.partial_ratio(a1,a2)/100 if a1 and a2 else 0,int(bool(a1 and a2) and a1==a2),jac(ta1,ta2),len(ta1&ta2),min(len(a1),len(a2))/max(1,len(a1),len(a2)),jac(da1,da2),int(bool(da1 and da2) and da1==da2),int(bool(da1 and da2) and not da1&da2),int(bool(p1 and p2 and p1&p2)),int(not a1),int(not a2),int(from_name),int(from_address),int(country1.casefold()==country2.casefold()),min(nr,ar),max(nr,ar),nr*ar,int(nr>.85 and ar<.3),int(ar>.85 and nr<.3)
    ],dtype=np.float32)

def rank_candidates(name,address,candidates,top_k=50):
    n,a=norm(name),norm(address)
    ranked=[]
    for row,nf,af in candidates:
        nn,aa=norm(row[1]),norm(row[2])
        ns=max(fuzz.ratio(n,nn),fuzz.token_set_ratio(n,nn))
        ads=max(fuzz.ratio(a,aa),fuzz.token_set_ratio(a,aa)) if a and aa else 0
        ranked.append((max(ns,ads),ns,ads,row,nf,af))
    ranked.sort(key=lambda x:(-x[0],-x[1],-x[2],x[3][0]))
    return [(x[3],x[4],x[5]) for x in ranked[:top_k]]
