"""Build disk-backed FTS5 indexes from the supplied TSV files."""
import sqlite3,time
from pathlib import Path
import pandas as pd

def build(data_root,split,index_path):
    data_root=Path(data_root);index_path=Path(index_path);index_path.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(index_path)
    con.execute('PRAGMA journal_mode=OFF');con.execute('PRAGMA synchronous=OFF')
    con.execute('PRAGMA cache_size=-100000');con.execute('PRAGMA temp_store=FILE')
    ready=con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='token_df'").fetchone()
    if ready and con.execute('SELECT COUNT(*) FROM token_df').fetchone()[0]>0:
        con.close();print(split,'index ready (cached)',flush=True);return
    con.execute('DROP TABLE IF EXISTS vocab')
    con.execute('DROP TABLE IF EXISTS token_df')
    existing=con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='targets'").fetchone()
    if existing:
        con.execute('DROP TABLE targets')
        con.commit()
    con.execute('CREATE VIRTUAL TABLE IF NOT EXISTS targets USING fts5(entity_id UNINDEXED,country,business_name,business_address,tokenize="unicode61 remove_diacritics 2")')
    n=0;start=time.time();cur=con.cursor()
    for source in (2,3):
        fn=data_root/split/f'{split}_source{source}.tsv'
        for chunk in pd.read_csv(fn,sep='\t',dtype=str,keep_default_na=False,chunksize=50000):
            cur.executemany('INSERT INTO targets(entity_id,country,business_name,business_address) VALUES(?,?,?,?)',((r.entity_id,r.country,r.business_name,r.business_address) for r in chunk.itertuples(index=False)))
            con.commit();n+=len(chunk)
            if n%500000<50000:print(split,'index rows',n,'seconds',round(time.time()-start),flush=True)
    con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(targets,'col')")
    con.execute('CREATE TABLE IF NOT EXISTS token_df(term TEXT,col TEXT,doc INTEGER,PRIMARY KEY(term,col)) WITHOUT ROWID')
    if con.execute('SELECT COUNT(*) FROM token_df').fetchone()[0]==0:
        con.execute('INSERT INTO token_df SELECT term,col,doc FROM vocab')
        con.commit()
    con.close();print(split,'index ready',n,flush=True)
