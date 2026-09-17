"""Learn Seoul district growth patterns and score comparable BC districts.

Scores are exploratory cross-source similarity scores, not calibrated BC
probabilities. Industry correspondence is provisional, explicitly recorded.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/transfer'
SEOUL = {'CS100001':'korean','CS100002':'chinese','CS100003':'japanese',
         'CS100004':'western','CS100005':'bakery','CS300001':'supermarket','CS300002':'convenience'}
BC = {8001:'korean',8002:'korean',8003:'korean',8005:'chinese',8004:'japanese',
      8006:'western',8301:'bakery',4020:'supermarket',4010:'convenience'}
CATS = list(SEOUL.values())
LABELS = {'korean':'한식','chinese':'중식','japanese':'일식','western':'양식','bakery':'제과점','supermarket':'슈퍼마켓','convenience':'편의점'}
FEATURES = ['amount_growth','count_growth','ticket_growth','concentration','q'] + [f'share_{c}' for c in CATS] + [f'share_change_{c}' for c in CATS]


def read_csv(path):
    for enc in ('utf-8-sig','cp949','euc-kr'):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            pass
    raise ValueError(f'Cannot decode {path}')


def panel(long):
    """Require all categories and contiguous quarters; never fill missing sales."""
    keys = ['region','period']
    a = long.pivot(index=keys, columns='category', values='amount').reindex(columns=CATS)
    n = long.pivot(index=keys, columns='category', values='count').reindex(columns=CATS)
    out = pd.DataFrame(index=a.index)
    out['amount'] = a.sum(axis=1, min_count=len(CATS))
    out['count'] = n.sum(axis=1, min_count=len(CATS))
    out['ticket'] = out.amount / out['count']
    for c in CATS:
        out[f'share_{c}'] = a[c] / out.amount
    out['concentration'] = (a.div(out.amount, axis=0)**2).sum(axis=1, min_count=len(CATS))
    out = out.reset_index().sort_values(keys).reset_index(drop=True)
    g = out.groupby('region', sort=False)
    contiguous = out.period.sub(g.period.shift()).eq(1)
    for col in ('amount','count','ticket'):
        out[f'{col}_growth'] = (out[col]/g[col].shift()-1).where(contiguous)
    for c in CATS:
        out[f'share_change_{c}'] = g[f'share_{c}'].diff().where(contiguous)
    out['year'] = out.period // 4
    out['q'] = out.period % 4 + 1
    out['quarter'] = out.year * 10 + out.q
    adjacent_future = g.period.shift(-1).sub(out.period).eq(1)
    out['future_growth'] = (g.amount.shift(-1)/out.amount-1).where(adjacent_future)
    out = out.replace([np.inf,-np.inf],np.nan)
    threshold = out.groupby('period').future_growth.transform(lambda s:s.quantile(.8))
    out['target'] = (out.future_growth >= threshold).astype(float).where(out.future_growth.notna())
    return out


def metrics(frame, scores, name, split):
    d = frame[['period','target']].copy()
    d['score'] = np.asarray(scores)
    top = d.groupby('period',group_keys=False).apply(lambda v:v.nlargest(max(1,math.ceil(len(v)*.2)),'score'))
    return {'model':name,'split':split,'rows':len(d),'positive_rate':d.target.mean(),
            'roc_auc':roc_auc_score(d.target,d.score), 'average_precision':average_precision_score(d.target,d.score),
            'top20_precision_by_quarter':top.target.mean()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    chunks=[]
    for path in sorted((ROOT/'data/raw/seoul/estimated_sales').glob('*.csv')):
        f=read_csv(path).iloc[:,:7].copy()
        f.columns=['quarter','admin','name','service','service_name','amount','count']
        f=f[f.service.isin(SEOUL)].copy()
        f['region']=f.admin.astype(str).str[:5]
        f['category']=f.service.map(SEOUL)
        f['period']=f.quarter//10*4+f.quarter%10-1
        chunks.append(f)
    s=pd.concat(chunks,ignore_index=True)
    assert not s.duplicated(['quarter','admin','service']).any(), 'Duplicate Seoul source keys'
    seoul=panel(s.groupby(['region','period','category'],as_index=False)[['amount','count']].sum())
    assert seoul.region.nunique()==25, 'Unexpected Seoul district codes'
    raw=read_csv(ROOT/'data/raw/raw_data.csv')
    raw['category']=raw.TP_BUZ_NO.map(BC)
    raw['region']=raw.SIDO_NM.str.strip()+' '+raw.CCG_NM.str.strip()
    raw['period']=raw.STRD_YYMM//100*4+(raw.STRD_YYMM%100-1)//3
    mapped=raw[raw.category.notna()].copy()
    # Retain unknown demographics: aggregate amount/count exactly once per row.
    keys=['STRD_YYMM','SIDO_NM','CCG_NM','GENDER_CD','AGE_CD','TP_BUZ_NO']
    assert not raw.duplicated(keys).any(), 'Duplicate BC demographic keys'
    monthly=mapped.groupby(['region','period','category']).STRD_YYMM.nunique()
    long=mapped.groupby(['region','period','category'],as_index=False)[['amt','cnt']].sum().rename(columns={'amt':'amount','cnt':'count'})
    long=long.merge(monthly.rename('months').reset_index(),on=['region','period','category'])
    long.loc[long.months.ne(3),['amount','count']]=np.nan
    bc=panel(long)
    labelled=seoul.dropna(subset=FEATURES+['target']).copy()
    train=labelled[labelled.quarter<=20233]
    val=labelled[labelled.quarter.between(20241,20243)]
    test=labelled[labelled.quarter.between(20251,20253)]
    # Hold out quarter before validation/test so training labels precede test features.
    assert train.period.max()+1 < val.period.min()
    estimators={
        'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,class_weight='balanced',random_state=42)),
        'random_forest':RandomForestClassifier(n_estimators=300,max_depth=4,min_samples_leaf=10,max_features=.7,class_weight='balanced',random_state=42,n_jobs=-1)}
    rows=[]
    fitted={}
    for name,est in estimators.items():
        est.fit(train[FEATURES],train.target)
        fitted[name]=est
        rows.append(metrics(val,est.predict_proba(val[FEATURES])[:,1],name,'validation'))
    for split,d in [('validation',val),('test',test)]:
        rows.append(metrics(d,d.amount_growth,'recent_growth_baseline',split))
    best=max([r for r in rows if r['split']=='validation' and r['model'] in estimators],key=lambda r:r['average_precision'])['model']
    importance=permutation_importance(fitted[best],val[FEATURES],val.target,scoring='average_precision',n_repeats=20,random_state=42,n_jobs=1)
    conditions=pd.DataFrame({'feature':FEATURES,'validation_importance':importance.importances_mean,'importance_std':importance.importances_std,
                             'growth_group_median':train[train.target.eq(1)][FEATURES].median().values,
                             'other_group_median':train[train.target.eq(0)][FEATURES].median().values}).sort_values('validation_importance',ascending=False)
    refit=labelled[labelled.quarter<=20243]
    assert refit.period.max()+1 < test.period.min()
    predictions=[]
    for name,est in estimators.items():
        model=clone(est).fit(refit[FEATURES],refit.target)
        score=model.predict_proba(test[FEATURES])[:,1]
        rows.append(metrics(test,score,name,'test'))
        predictions.append(test[['region','quarter','target','future_growth']].assign(model=name,score=score))
    final=clone(estimators[best]).fit(labelled[FEATURES],labelled.target)
    latest=bc[bc.period.eq(bc.period.max())].copy()
    latest['missing_features']=latest[FEATURES].isna().sum(axis=1)
    excluded=latest[latest.missing_features.gt(0)].copy()
    candidates=latest[latest.missing_features.eq(0)].copy()
    candidates['similarity_score']=100*final.predict_proba(candidates[FEATURES])[:,1]
    outside=(candidates[FEATURES].lt(labelled[FEATURES].min()) | candidates[FEATURES].gt(labelled[FEATURES].max()))
    candidates['out_of_range_features']=outside.apply(lambda r:', '.join(r.index[r]),axis=1)
    candidates['out_of_range_count']=outside.sum(axis=1)
    candidates['interpretation']='탐색용: 검증 적중률이 무작위 기준을 넘지 못함'
    contributions=[]
    if best=='logistic':
        scaled=final.named_steps['standardscaler'].transform(candidates[FEATURES])
        effects=scaled*final.named_steps['logisticregression'].coef_[0]
        for pos,(_,r) in enumerate(candidates.iterrows()):
            for j,col in enumerate(FEATURES):
                contributions.append({'region':r.region,'feature':col,'value':r[col],'log_odds_contribution':effects[pos,j]})
        candidates['positive_score_drivers']=[
            '; '.join(f'{FEATURES[j]}={effects[i,j]:+.3f}' for j in np.argsort(effects[i])[::-1][:3] if effects[i,j]>0)
            for i in range(len(candidates))]
        candidates['negative_score_drivers']=[
            '; '.join(f'{FEATURES[j]}={effects[i,j]:+.3f}' for j in np.argsort(effects[i])[:3] if effects[i,j]<0)
            for i in range(len(candidates))]
    for col in FEATURES:
        candidates[f'percentile_{col}']=candidates[col].map(lambda v:float(labelled[col].le(v).mean()))
    candidates['evidence']=candidates.apply(lambda r:f"매출 전분기 대비 {r.amount_growth:+.1%}; 건수 {r.count_growth:+.1%}; 건당금액 {r.ticket_growth:+.1%}; 최대 업종 {LABELS[max(CATS,key=lambda c:r['share_'+c])]}",axis=1)
    candidates=candidates.sort_values('similarity_score',ascending=False).reset_index(drop=True)
    candidates.insert(0,'rank',np.arange(1,len(candidates)+1))
    for name,df in [('seoul_features',seoul),('bc_features',bc),('candidates',candidates),('excluded_regions',excluded),('conditions',conditions),('metrics',pd.DataFrame(rows)),('test_predictions',pd.concat(predictions))]:
        df.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'bc_code':k,'category':v,'seoul_code':next(c for c,n in SEOUL.items() if n==v),'status':'provisional_name_based'} for k,v in BC.items()]).to_csv(OUT/'industry_mapping.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(contributions).to_csv(OUT/'candidate_contributions.csv',index=False,encoding='utf-8-sig')
    joblib.dump({'model':final,'features':FEATURES,'bc_mapping':BC,'seoul_mapping':SEOUL},OUT/'model.joblib')
    audit={'selected_model':best,'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),'final_fit_rows':len(labelled),
           'bc_regions_scored':len(candidates),'bc_regions_excluded':len(excluded),'bc_unmapped_industries':sorted(raw.loc[raw.category.isna(),'TP_BUZ_NO'].unique().tolist()),
           'unmapped_amount_share':float(raw.loc[raw.category.isna(),'amt'].sum()/raw.amt.sum()),'forecast_quarter':'2026Q3',
           'target':'Next-quarter matched-category sales growth top 20% across Seoul districts',
           'limitations':['Industry correspondence requires provider codebook confirmation','Seoul estimated sales and BC coverage differ','No BC future outcomes to validate transfer','25 Seoul districts only; quarterly observations correlated','Relative growth label is not a confirmed hotplace label','Two BC quarters cannot provide acceleration or long-term trends','District code uses first five digits; boundary history not independently verified','Missing category sales not treated as zero; incomplete regions excluded','Score is not a calibrated BC probability']}
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# 서울시 성장 패턴 기반 BC카드 후보 탐색','',
            '**검증에서 선택 모델의 상위 20% 적중률은 20%로 무작위 기준과 같습니다. 후보 순위는 탐색용이며 유효한 전국 핫플 예측으로 검증되지 않았습니다.**',
            '2026년 1~6월 BC카드 소비로 2026년 3분기 성장 후보를 탐색합니다. 점수는 서울 패턴 유사도이며 BC카드 미래 성장 확률로 검증되지 않았습니다.',
            '서울시 행정동을 구 단위로 집계했습니다. 공통 7개 업종의 매출·건수 증가율, 건당금액 증가율, 업종 비중·변화 및 집중도와 분기를 사용합니다.',
            '대형할인점·스넥은 명확한 일대일 대응이 없어 제외했습니다. 한식은 BC 일반한식·갈비전문점·한정식을 합친 잠정 대응입니다.',
            '2021~2023Q3 학습 / 2024Q1~Q3 검증. 검증 AP로 모델 선택 후 2024Q3까지 재학습하여 2025Q1~Q3 테스트. 경계 분기를 제외하여 미래 정답의 시간 중첩을 피했습니다.',
            '최종 후보 점수는 알려진 서울시 정답 전체로 재학습한 모델로 계산합니다. 테스트 성능은 재학습 전 별도 모델 결과입니다.',
            f'선택 모델: {best}. 후보 {len(candidates)}개, 입력 부족으로 제외 {len(excluded)}개.',
            '조건표의 중요도는 검증 AP 감소량이고, 양수인 변수만 예측 기여 근거가 있습니다. 그룹 중앙값과 후보 관측 변화는 인과적 핫플 조건이 아닙니다.',
            'candidate_contributions.csv는 학습 평균을 기준으로 표준화한 변수값 × 로지스틱 계수입니다. 양수는 점수를 높이는 방향이고 확률 변화량은 아닙니다. 업종 비중과 성장률 간 상관관계 때문에 독립적인 인과 효과로 해석할 수 없습니다.',
            'Top20 Precision은 각 분기 안에서 상위 20%를 고른 적중률입니다. AP는 average precision 정의입니다.',
            '원본 한글은 정상이며 과거 깨진 출력은 터미널 인코딩 문제였습니다.',
            '', '## 평가',pd.DataFrame(rows).to_csv(index=False),'## 상위 후보',candidates[['rank','region','similarity_score','out_of_range_count','evidence']].head(20).to_csv(index=False),
            '## 해석 제한',*audit['limitations'],'','재실행: `uv run python src/transfer_candidates.py`']
    (OUT/'report.md').write_text('\n\n'.join(report),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(candidates[['rank','region','similarity_score','out_of_range_count','evidence']].head(10).to_string(index=False))
    print(conditions.head(8).to_string(index=False))


if __name__=='__main__':
    main()
