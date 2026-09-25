"""SQLite FTS5 retrieval, using only supplied source records."""
import re
import sqlite3
import unicodedata
from functools import lru_cache

NAME_STOP={'inc','incorporated','ltd','limited','llc','private','pvt','co','company','corp','corporation','the','and','of','sarl','sas','sci','sa','privee','entreprise'}
ADDRESS_STOP={'road','rd','street','st','avenue','ave','lane','ln','city','district','state','india','usa','united','near','block','floor','no','number','west','east','north','south','rue','de','du','des','la','le','les','france','texas','delhi','maharashtra','karnataka','california'}

def tokens(s):
    s=unicodedata.normalize('NFKD',str(s).casefold())
    s=''.join(c for c in s if not unicodedata.combining(c))
    return re.findall(r'[^\W_]+',s,flags=re.UNICODE)

@lru_cache(maxsize=20000)
def norm(s):
    return ' '.join(tokens(s))

def choose(ts,stop):
    return sorted({t for t in ts if len(t)>=3 and t not in stop},key=lambda t:(-len(t),t))[:4]

def rare(con,ts,column,limit=5):
    out=[]
    for t in set(ts):
        if len(t)<3:continue
        r=con.execute('SELECT doc FROM token_df WHERE term=? AND col=?',(t,column)).fetchone()
        if r:out.append((r[0],t))
    return sorted(out)[:limit]

def query_fts(con,country,column,terms,k):
    if not terms: return []
    terms=[t.replace('"','') for t in terms]
    country=country.replace('"','')
    expr=f'country:"{country}" AND {column}:('+ ' AND '.join(f'"{t}"' for t in terms)+')'
    try:
        return con.execute('SELECT entity_id,business_name,business_address,country FROM targets WHERE targets MATCH ? LIMIT ?', (expr,k)).fetchall()
    except sqlite3.OperationalError as exc:
        raise RuntimeError(f'FTS query failed: {expr}') from exc

def retrieve(con,name,address,country,k_name=20,k_address=20,extra=False):
    nt=rare(con,[t for t in tokens(name) if t not in NAME_STOP],'business_name')
    at=rare(con,[t for t in tokens(address) if t not in ADDRESS_STOP],'business_address')
    nums=[t for t in tokens(address) if any(c.isdigit() for c in t) and len(t)>=2]
    qs=[]
    if len(nt)>=2: qs.append(('business_name',[nt[0][1],nt[1][1]],k_name))
    if len(nt)>=3: qs.append(('business_name',[nt[0][1],nt[2][1]],k_name))
    if len(at)>=2: qs.append(('business_address',[at[0][1],at[1][1]],k_address))
    if len(at)>=3: qs.append(('business_address',[at[0][1],at[2][1]],k_address))
    if nums and at: qs.append(('business_address',[nums[0],at[0][1]],k_address))
    if nt and nt[0][0]<=20000: qs.append(('business_name',[nt[0][1]],k_name))
    if len(nt)>=2 and nt[1][0]<=20000: qs.append(('business_name',[nt[1][1]],k_name))
    if at and at[0][0]<=20000: qs.append(('business_address',[at[0][1]],k_address))
    if len(at)>=2 and at[1][0]<=20000: qs.append(('business_address',[at[1][1]],k_address))
    found={}
    for col,terms,k in qs:
        for row in query_fts(con,country,col,terms,k):
            item=found.setdefault(row[0],[row,0,0])
            if col=='business_name':item[1]=1
            else:item[2]=1
    return list(found.values())

def connect(path):
    con=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    con.execute('PRAGMA cache_size=-100000')
    return con
