import numpy as np, pandas as pd
from early_warning_scores import *
from evaluate_ews_baseline import evaluate
assert [news2_rr(x) for x in [8,9,11,12,20,21,24,25]]==[3,1,1,0,0,2,2,3]
assert [news2_spo2_scale1(x) for x in [91,92,93,94,95,96]]==[3,2,2,1,1,0]
assert [news2_sbp(x) for x in [90,91,100,101,110,111,219,220]]==[3,2,2,1,1,0,0,3]
assert [news2_hr(x) for x in [40,41,50,51,90,91,110,111,130,131]]==[3,1,1,0,0,1,1,2,2,3]
assert [news2_temp(x) for x in [35.0,35.1,36.0,36.1,38.0,38.1,39.0,39.1]]==[3,1,1,0,0,1,1,2]
assert [news2_consciousness(g) for g in [15,14,3]]==[0,3,3]
assert [news2_spo2_scale2(x,o) for x,o in [(83,0),(85,0),(87,0),(92,0),(97,0),(94,1),(96,1),(97,1)]]==[3,2,1,0,0,1,2,3]
assert [mews_sbp(x) for x in [70,71,80,81,100,101,199,200]]==[3,2,2,1,1,0,0,2]
assert [mews_hr(x) for x in [40,41,50,51,100,101,110,111,129,130]]==[2,1,1,0,0,1,1,2,2,3]
assert [mews_rr(x) for x in [8,9,14,15,20,21,29,30]]==[2,0,0,1,1,2,2,3]
assert [mews_temp(x) for x in [34.9,35.0,38.4,38.5]]==[2,0,0,2]
assert [mews_avpu(g) for g in [15,13,10,8]]==[0,1,2,3]
r={"SBP":120,"HR":75,"RR":16,"BT":36.8,"SpO2":98,"GCS":15}
assert news2_row(r)==0 and mews_row(r)==1
rng=np.random.default_rng(1); rows=[]
for p in range(112):
    lab=int(p<19)
    for k in range(rng.integers(5,30)):
        rows.append(dict(patient_id=p,label=lab,SBP=rng.normal(115-15*lab,15),HR=rng.normal(85+20*lab,12),
          RR=rng.normal(18+4*lab,3),BT=rng.normal(36.8,.5),SpO2=min(100,rng.normal(96-2*lab,2)),GCS=15 if rng.random()>.1 else 13))
df=pd.DataFrame(rows); df.loc[rng.random(len(df))<.05,"RR"]=np.nan
sc=patient_scores(df); y=df.groupby("patient_id",sort=False).label.max().loc[sc.index].values
pf,s,t=evaluate(sc,y,seed=42); print(s.round(3)); print(t.round(3)); print(report_missing(df))
print("ALL TESTS PASSED")
