"""Structural features: stand pier/number, destination region. Helps generalize to unseen stands."""
from config import DATA_DIR, WORK_DIR, RANKING
import pandas as pd, numpy as np, pickle, re, pyarrow.parquet as pq
OUT=(WORK_DIR + "/")
vocab=pickle.load(open(OUT+"vocab.pkl","rb"))
d=pq.read_table(OUT+"dep.parquet",columns=["STAND_mvt","ADES_mvt","AIRCRAFT_TYPE_mvt"]).to_pandas()
st=list(vocab["STAND_mvt"]); ad=list(vocab["ADES_mvt"])

pref=np.full(len(st),-1,np.int32); num=np.full(len(st),np.nan,np.float32)
plen=np.full(len(st),np.nan,np.float32); pmap={}
for i,s in enumerate(st):
    s=str(s); m=re.match(r"^([A-Za-z]*)",s); p=m.group(1).upper() if m else ""
    if p not in pmap: pmap[p]=len(pmap)
    pref[i]=pmap[p]
    dg="".join(ch for ch in s if ch.isdigit())
    num[i]=float(dg[:4]) if dg else np.nan
    plen[i]=len(s)
reg=np.full(len(ad),-1,np.int32); rmap={}
for i,s in enumerate(ad):
    r=str(s)[:2].upper()
    if r not in rmap: rmap[r]=len(rmap)
    reg[i]=rmap[r]
cn=np.full(len(ad),-1,np.int32); cmap={}
for i,s in enumerate(ad):
    r=str(s)[:1].upper()
    if r not in cmap: cmap[r]=len(cmap)
    cn[i]=cmap[r]

sv=d.STAND_mvt.values; av=d.ADES_mvt.values
def gather(tbl,codes,fill=-1):
    out=np.full(len(codes),fill,dtype=tbl.dtype)
    ok=codes>=0; out[ok]=tbl[codes[ok]]; return out
res={"stand_pier":gather(pref,sv).astype(np.float32),
     "stand_num":gather(num,sv,np.nan).astype(np.float32),
     "stand_len":gather(plen,sv,np.nan).astype(np.float32),
     "ades_region":gather(reg,av).astype(np.float32),
     "ades_cont":gather(cn,av).astype(np.float32)}
pd.DataFrame(res).to_parquet(OUT+"f6.parquet",index=False)
print("f6 saved; piers",len(pmap),"regions",len(rmap),flush=True)
