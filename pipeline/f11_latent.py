"""Latent airport geometry by matrix factorization.

There are no coordinates in this dataset, but the matrix
      M[stand, runway] = mean taxi-proxy
is a distance-like table. Truncated SVD of M yields low-rank embeddings that behave
like latent positions: stands that taxi similarly to all runways get similar vectors.
This generalises to rare stand-runway pairs where a target encoding has no support.

Matrices are built from TRAINING rows only, then applied to every row.
"""
from config import DATA_DIR, WORK_DIR, RANKING
import numpy as np, pandas as pd, gc, pickle
import pyarrow.parquet as pq
from sklearn.decomposition import TruncatedSVD
OUT=(WORK_DIR + "/")
K=4
d=pq.read_table(OUT+"dep.parquet",columns=["APT","STAND_mvt","RUNWAY_mvt","ADES_mvt",
                                           "AIRCRAFT_OPERATOR_flt","is_rank"]).to_pandas()
f1=pq.read_table(OUT+"f1.parquet",columns=["d_aobt"]).to_pandas()
f6=pq.read_table(OUT+"f6.parquet",columns=["stand_pier","ades_region"]).to_pandas()
n=len(d); tr=~d.is_rank.values
prox=f1.d_aobt.values.astype(np.float64)
apt=d.APT.values.astype(np.int64)
cols={"stand":d.STAND_mvt.values.astype(np.int64),"rwy":d.RUNWAY_mvt.values.astype(np.int64),
      "pier":np.nan_to_num(f6.stand_pier.values,nan=-1).astype(np.int64),
      "reg":np.nan_to_num(f6.ades_region.values,nan=-1).astype(np.int64),
      "oper":d.AIRCRAFT_OPERATOR_flt.values.astype(np.int64)}
del d,f1,f6; gc.collect()

def factorize(rowkey,colkey,tag,k=K,minc=5):
    """SVD of the row x col mean-proxy matrix; emit row and col embeddings + reconstruction."""
    out={f"{tag}_r{i}":np.full(n,np.nan,np.float32) for i in range(k)}
    out.update({f"{tag}_c{i}":np.full(n,np.nan,np.float32) for i in range(k)})
    out[f"{tag}_rec"]=np.full(n,np.nan,np.float32)
    for a in np.unique(apt):
        m=apt==a; mt=m&tr&np.isfinite(prox)
        if mt.sum()<500: continue
        r=rowkey[mt]; c=colkey[mt]; v=prox[mt]
        ru=np.unique(r); cu=np.unique(c)
        ri={x:i for i,x in enumerate(ru)}; ci={x:i for i,x in enumerate(cu)}
        S=np.zeros((len(ru),len(cu))); C=np.zeros((len(ru),len(cu)))
        np.add.at(S,( [ri[x] for x in r],[ci[x] for x in c]),v)
        np.add.at(C,( [ri[x] for x in r],[ci[x] for x in c]),1.0)
        M=np.where(C>=minc,S/np.maximum(C,1),np.nan)
        gm=np.nanmean(M)
        rm=np.nanmean(M,axis=1); cm=np.nanmean(M,axis=0)
        rm=np.where(np.isfinite(rm),rm,gm); cm=np.where(np.isfinite(cm),cm,gm)
        # centre then impute, so the SVD models interaction (geometry), not marginals
        Mc=M-rm[:,None]-cm[None,:]+gm
        Mc=np.where(np.isfinite(Mc),Mc,0.0)
        kk=min(k,min(Mc.shape)-1)
        if kk<1: continue
        sv=TruncatedSVD(n_components=kk,random_state=0)
        U=sv.fit_transform(Mc); V=sv.components_.T
        R=U@V.T+rm[:,None]+cm[None,:]-gm
        rr=np.array([ri.get(x,-1) for x in rowkey[m]]); cc=np.array([ci.get(x,-1) for x in colkey[m]])
        idx=np.where(m)[0]
        okr=rr>=0; okc=cc>=0
        for i in range(kk):
            out[f"{tag}_r{i}"][idx[okr]]=U[rr[okr],i].astype(np.float32)
            out[f"{tag}_c{i}"][idx[okc]]=V[cc[okc],i].astype(np.float32)
        both=okr&okc
        out[f"{tag}_rec"][idx[both]]=R[rr[both],cc[both]].astype(np.float32)
    return out

R={}
R.update(factorize(cols["stand"],cols["rwy"],"svd_sr"))     # stand x runway  -> latent geometry
R.update(factorize(cols["pier"], cols["rwy"],"svd_pr"))     # pier  x runway  -> coarse, robust
R.update(factorize(cols["stand"],cols["reg"],"svd_sg"))     # stand x route direction (SID proxy)
R.update(factorize(cols["oper"], cols["rwy"],"svd_or"))     # operator x runway -> behaviour
pd.DataFrame(R).to_parquet(OUT+"f11.parquet",index=False)
print("f11 saved",len(R),"features:",sorted(R.keys())[:6],"...",flush=True)
