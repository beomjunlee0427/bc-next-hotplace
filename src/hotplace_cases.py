"""Descriptive Seoul case study, with baseline-only matched comparisons.

No model probabilities or national rankings. The observation window cannot
establish the original emergence date of long-established hotspots.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transfer_candidates import read_csv, ROOT

OUT=ROOT/'reports/hotplace_cases'
CASES={'성수1가1동':'성수','성수1가2동':'성수','성수2가1동':'성수','성수2가3동':'성수','연남동':'연남','한강로동':'용리단길'}
MATCH=['log_amount','log_count','cafe_share','young_share','weekend_share','log_stores']
LEVELS=['amount','count','stores','population','cafe_share','young_share','weekend_share','opening_rate','closure_rate','western_share_common','bakery_share_common']
SOURCES=[
    ('성수 행정동별 관광지 위치 참고','https://map.sd.go.kr/trip_new/tMap.jsp'),
    ('연남동 경의선숲길 2015년 변화','https://love.seoul.go.kr/articles/10581'),
    ('용산구 관광사업 활성화 연구 2024, 용리단길 2022년 이후 매출 증가','https://yscl.go.kr/kr/bbs/download.do?bbs_id=notice&uid=B013E768463C125104574D1CE7AF6521')]


def build():
    chunks=[]
    for path in sorted((ROOT/'data/raw/seoul/estimated_sales').glob('*.csv')):
        f=read_csv(path)
        f=f[f['서비스_업종_코드'].str.startswith('CS100')].copy()
        f=f.rename(columns={'기준_년분기_코드':'quarter','행정동_코드':'admin','행정동_코드_명':'name',
                            '서비스_업종_코드':'industry','당월_매출_금액':'amount','당월_매출_건수':'count'})
        f['cafe_amount']=f.amount.where(f.industry.eq('CS100010'),0)
        common=f.industry.isin(['CS100001','CS100002','CS100003','CS100004','CS100005'])
        f['common_amount']=f.amount.where(common,0)
        f['western_amount']=f.amount.where(f.industry.eq('CS100004'),0)
        f['bakery_amount']=f.amount.where(f.industry.eq('CS100005'),0)
        agecols=[c for c in f if c.startswith('연령대_') and c.endswith('_매출_금액')]
        assert len(agecols)==6
        f['known_age_amount']=f[agecols].sum(axis=1,min_count=6)
        f['young_amount']=f['연령대_20_매출_금액']+f['연령대_30_매출_금액']
        f['weekend_amount']=f['주말_매출_금액']
        chunks.append(f[['quarter','admin','name','industry','amount','count','cafe_amount','common_amount','western_amount','bakery_amount','known_age_amount','young_amount','weekend_amount']])
    sales=pd.concat(chunks,ignore_index=True)
    assert not sales.duplicated(['quarter','admin','industry']).any()
    quantities=list(sales.columns[4:])
    p=sales.groupby(['quarter','admin','name'],as_index=False)[quantities].sum(min_count=1)
    stores=[]
    for path in sorted((ROOT/'data/raw/seoul/store').glob('*행정동*.csv')):
        f=read_csv(path)
        f=f[f['서비스_업종_코드'].str.startswith('CS100')].rename(columns={
            '기준_년분기_코드':'quarter','행정동_코드':'admin','점포_수':'stores','개업_점포_수':'openings','폐업_점포_수':'closures'})
        stores.append(f[['quarter','admin','stores','openings','closures']])
    st=pd.concat(stores).groupby(['quarter','admin'],as_index=False).sum(min_count=1)
    pop=read_csv(next((ROOT/'data/raw/seoul/population_consumption').glob('*길단위*'))).rename(columns={
        '기준_년분기_코드':'quarter','행정동_코드':'admin','총_유동인구_수':'population'})
    assert not pop.duplicated(['quarter','admin']).any()
    p=p.merge(st,on=['quarter','admin'],how='left',validate='one_to_one').merge(pop[['quarter','admin','population']],on=['quarter','admin'],how='left',validate='one_to_one')
    return p


def ratios(p):
    p=p.copy()
    for dest,num,den in [('cafe_share','cafe_amount','amount'),('young_share','young_amount','known_age_amount'),
                         ('weekend_share','weekend_amount','amount'),('opening_rate','openings','stores'),('closure_rate','closures','stores'),
                         ('western_share_common','western_amount','common_amount'),('bakery_share_common','bakery_amount','common_amount')]:
        p[dest]=p[num]/p[den].replace(0,np.nan)
    return p.replace([np.inf,-np.inf],np.nan)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p=ratios(build()).sort_values(['admin','quarter']).reset_index(drop=True)
    # Restrict comparisons to stable observed codes, all twenty quarters.
    stable=p.groupby('admin').quarter.nunique().eq(20)
    base=p[p.quarter.between(20211,20214)].groupby(['admin','name'],as_index=False).mean(numeric_only=True)
    for c in ('amount','count','stores'):
        base['log_'+c]=np.log1p(base[c])
    base=base[base.admin.isin(stable.index[stable])].dropna(subset=MATCH).set_index('admin')
    # Avoid using other known visitor destinations as presumed non-hot controls.
    excluded=base.name.str.contains('성수|연남|한강로|서교|합정|망원|한남|이태원|종로1|삼청|가회|청운효자|사직|을지로|신사|압구정|명동|회현|신촌|문래|성내2',regex=True)
    pool=base[~excluded]
    scale=pool[MATCH].std().replace(0,1)
    matched=[]
    for admin,r in base[base.name.isin(CASES)].iterrows():
        dist=(((pool[MATCH]-r[MATCH].astype(float))/scale)**2).sum(axis=1)**.5
        for rank,(other,d) in enumerate(dist.nsmallest(10).items(),1):
            matched.append({'case_admin':admin,'case_name':r['name'],'case_group':CASES[r['name']],
                            'control_admin':other,'control_name':base.loc[other,'name'],'distance':d,'rank':rank})
    all_matches=pd.DataFrame(matched)
    match=all_matches[all_matches['rank'].le(5)].copy()
    assert set(match.case_name)==set(CASES)
    balances=[]
    for case,m in match.groupby('case_admin'):
        for col in MATCH:
            value=base.loc[case,col]; control=base.loc[m.control_admin,col].mean()
            balances.append({'case_admin':case,'case_name':base.loc[case,'name'],'feature':col,'case_baseline':value,'control_baseline':control,'standardized_difference':(value-control)/scale[col]})
    numeric=['amount','count','cafe_amount','common_amount','western_amount','bakery_amount','known_age_amount','young_amount','weekend_amount','stores','openings','closures','population']
    trajectories=[]
    for group in sorted(set(CASES.values())):
        case_ids=match.loc[match.case_group.eq(group),'case_admin'].unique()
        control_ids=match.loc[match.case_group.eq(group),'control_admin'].unique()
        for side,ids in [('case',case_ids),('comparison',control_ids)]:
            v=ratios(p[p.admin.isin(ids)].groupby('quarter',as_index=False)[numeric].sum(min_count=1))
            v['group']=group;v['side']=side
            for c in ('amount','count','stores','population'):
                v[c+'_index']=100*v[c]/v.loc[v.quarter.between(20211,20214),c].mean()
                v[c+'_yoy']=v[c].pct_change(4,fill_method=None)
            trajectories.append(v)
    traj=pd.concat(trajectories,ignore_index=True)
    summaries=[]
    for group,t in traj.groupby('group'):
        for col in LEVELS:
            vals={}
            for side,d in t.groupby('side'):
                early=d[d.quarter.between(20211,20214)][col].mean()
                late=d[d.quarter.between(20251,20254)][col].mean()
                vals[side]=(late/early-1) if col in ['amount','count','stores','population'] else (late-early)
            summaries.append({'group':group,'metric':col,'unit':'growth' if col in ['amount','count','stores','population'] else 'share_change',
                              'case_change':vals['case'],'comparison_change':vals['comparison'],'difference':vals['case']-vals['comparison']})
    summary=pd.DataFrame(summaries)
    sensitivity=[]
    individual=[]
    for k in (3,5,10):
        mk=all_matches[all_matches['rank'].le(k)]
        for group,m in mk.groupby('case_group'):
            results={}
            for side,ids in [('case',m.case_admin.unique()),('comparison',m.control_admin.unique())]:
                d=ratios(p[p.admin.isin(ids)].groupby('quarter',as_index=False)[numeric].sum(min_count=1))
                early=d[d.quarter.between(20211,20214)][LEVELS].mean()
                late=d[d.quarter.between(20251,20254)][LEVELS].mean()
                change=late-early
                for c in ('amount','count','stores','population'):
                    change[c]=late[c]/early[c]-1
                results[side]=change
            for metric in LEVELS:
                sensitivity.append({'group':group,'controls_per_dong':k,'metric':metric,'difference':results['case'][metric]-results['comparison'][metric]})
    for name in CASES:
        d=p[p.name.eq(name)]
        for metric in LEVELS:
            early=d[d.quarter.between(20211,20214)][metric].mean()
            late=d[d.quarter.between(20251,20254)][metric].mean()
            individual.append({'name':name,'metric':metric,'value_2021':early,'value_2025':late,'change':late/early-1 if metric in ['amount','count','stores','population'] else late-early})
    # Lead checks use within-quarter case-minus-comparison YoY changes.
    # Correlation is descriptive; no causal or independent-sample p-values.
    leads=[]
    signals=['population_yoy','stores_yoy','cafe_share','young_share','weekend_share','opening_rate','western_share_common','bakery_share_common']
    for group,t in traj.groupby('group'):
        a=t[t.side.eq('case')].set_index('quarter').sort_index()
        b=t[t.side.eq('comparison')].set_index('quarter').sort_index()
        outcome=(a.amount_yoy-b.amount_yoy).shift(-1)
        for col in signals:
            x=a[col]-b[col]
            if col not in ['population_yoy','stores_yoy']:
                x=x.diff(4)
            paired=pd.concat([x.rename('x'),outcome.rename('next_yoy_excess')],axis=1).dropna()
            leads.append({'group':group,'signal':col,'paired_quarters':len(paired),'correlation':paired.x.corr(paired.next_yoy_excess)})
    lead=pd.DataFrame(leads)
    transfer=pd.DataFrame([
        ['取引件数・매출 증가','amt / cnt','분기 대비 계산 가능; 전년 대비 검증에는 과거 BC 필요'],
        ['양식·제과 비중 변화','TP_BUZ_NO, amt','공통 음식업 5종 내 비중으로 잠정 비교 가능; 공식 업종 대응 확인 필요'],
        ['카페 비중','없음','현재 BC 업종에 카페 없음'],
        ['20~30대 소비 비중','AGE_CD','코드북과 연령 구간 확인 전 적용 불가'],
        ['주말 소비 비중','없음','현재 BC 월별 집계로 적용 불가'],
        ['유동인구·개업·폐업','없음','별도 외부 자료 필요']
    ],columns=['signal','bc_columns','availability'])
    transfer.loc[0,'signal']='거래 건수·매출 증가'
    mapping=match[['case_group','case_name','case_admin']].drop_duplicates()
    mapping['scope']='행정동 대리 범위; 명명된 핫플 골목 경계와 일치하지 않음'
    for name,df in [('panel',p),('case_mapping',mapping),('matched_controls',match),('matching_balance',pd.DataFrame(balances)),('trajectories',traj),('changes_2021_2025',summary),('lead_checks',lead),('bc_transferability',transfer),('control_sensitivity',pd.DataFrame(sensitivity)),('individual_dong_changes',pd.DataFrame(individual))]:
        df.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(3,3,figsize=(15,11),sharex=True)
    for row,group in enumerate(sorted(set(CASES.values()))):
        for col,(metric,title) in enumerate([('amount_index','음식업 매출 지수'),('count_index','음식업 거래 건수 지수'),('cafe_share','음식업 내 카페 매출 비중')]):
            ax=axs[row,col]
            for side,label in [('case','사례'),('comparison','2021년 유사 비교 지역')]:
                d=traj[(traj.group==group)&(traj.side==side)].sort_values('quarter')
                ax.plot(np.arange(len(d)),d[metric]*(100 if metric=='cafe_share' else 1),label=label)
            ax.set_title(group+' · '+title);ax.grid(alpha=.2)
            ax.set_xticks([0,4,8,12,16],['2021Q1','2022Q1','2023Q1','2024Q1','2025Q1'])
            ax.set_ylabel('%' if metric=='cafe_share' else '2021 평균=100')
            if row==0 and col==0:ax.legend(fontsize=8)
    fig.suptitle('서울 핫플 행정동 사례: 성장 궤적 비교 (발생 시점·인과효과 추정 아님)')
    fig.tight_layout();fig.savefig(OUT/'case_trends.png',dpi=150);plt.close(fig)
    report=['# 서울 핫플 사례의 관측 가능한 성장 특징','',
        '## 이번 분석에서 확인한 내용',
        '한강로동은 2021 대비 2025 음식업 매출 +81.2%, 건수 +25.8%, 점포수 +37.2%입니다. 비교 지역은 매출 +31.9%, 건수 +4.3%입니다. 주말 소비 비중 +8.83%p와 20~30대 비중 +4.93%p가 함께 관측됐습니다. 용리단길 골목만의 수치가 아니라 한강로동 전체입니다.',
        '성수 4개 동 합계는 매출 +35.3%, 건수 +5.8%로 비교군(+32.3%, +5.8%)과 음식업 성장 차이가 크지 않습니다. 연남동은 매출 -1.9%, 건수 -15.6%로 이번 기간의 지속 성장 사례로 적합하지 않습니다.',
        '20~30대 비중 변화는 세 사례 모두 비교군보다 높았습니다(성수 +4.51%p, 연남 +3.74%p, 한강로 +10.86%p 차이). 다만 성수·연남의 실제 비중은 각각 -1.79%p, -0.45%p로 감소했습니다. 청년층 비중 증가라는 공통 조건으로 해석하면 안 됩니다.',
        '카페 매출 비중은 세 사례 모두 2021보다 낮았습니다. 따라서 카페 비중의 단순 증가를 필수 조건으로 삼을 근거가 없습니다. 매출 비중 하락이 카페 매출액 하락을 뜻하지도 않습니다.',
        '비교 동을 3/5/10개로 변경해도 한강로의 상대적 매출·건수 성장과 세 사례의 상대적 청년 비중 유지 방향은 같았습니다. 매칭 잔여 차이는 일부 변수에서 약 1.07 표준편차로 커서 동일 조건의 대조군이라고 단정할 수 없습니다.',
        '분석 범위: 2021Q1~2025Q4, 서울시 음식업 10종. 성수 4개 행정동·연남동·한강로동을 사례로 사용합니다. 성수는 넓은 행정동 합계이며 용리단길은 한강로동 전체의 대리 관측입니다. 골목 경계 매핑을 완료한 결과가 아닙니다.',
        '연남동의 초기 성장은 관측 기간 이전입니다. 용리단길 관련 의회 보고서는 2022년 이후 매출 증가를 다루지만 정확한 발생 분기를 임의로 지정하지 않았습니다. 이 분석은 초기 발생 전후의 인과 분석이 아닙니다.',
        '## 비교 방법',
        '2021년 평균 매출·건수·점포수(로그), 카페 비중, 알려진 연령 매출 중 20~30대 비중, 주말 매출 비중을 표준화하여 사례 행정동별 가장 가까운 5개 동을 선정했습니다. 대상·알려진 주요 관광상권 이름은 비교 풀에서 제외했습니다. 비핫플로 공식 검증된 대조군은 아니며 비교 지역 중복은 그룹 합산 시 제거했습니다.',
        '20개 분기가 모두 있는 행정동만 비교 후보로 사용합니다. matching_balance.csv의 잔여 불균형을 함께 보아야 합니다. 비교군 선정에 2022년 이후 수치값은 사용하지 않았지만 전체 기간 자료 존재 조건은 생존 편향을 만들 수 있습니다.',
        '2021년은 코로나 시기입니다. 비교군을 함께 보아도 코로나·지역별 회복 차이나 관광 계절성을 완전히 제거하지 못합니다. 2025/2021 비교는 인과효과가 아닙니다.',
        '## 주요 변화: 2021년 분기 평균 대비 2025년 분기 평균',summary.to_csv(index=False),
        '## 한 분기 선행 관계 탐색',
        '지역-비교군의 전년 대비 변화 차이를 다음 분기 매출 전년 대비 증가율 차이와 비교합니다. 표본은 사례당 최대 15분기이며 연속 시점이 독립이 아니므로 상관계수를 유의한 선행 조건으로 단정하지 않습니다. 여러 변수 탐색에 따른 우연한 상관 및 매출 구성비의 분모 효과도 있습니다.',lead.to_csv(index=False),
        '전년 대비 시계열의 인접 시점은 공통 기간·추세를 공유합니다. 단순 한 분기 선행 상관은 그 영향을 제거하지 못하며 예측의 추가 가치를 검증한 결과가 아닙니다.',
        'control_sensitivity.csv는 비교 동 3/5/10개 선택 민감도이고 individual_dong_changes.csv는 넓은 성수 합계에 가려지는 개별 동 변화를 보여줍니다.',
        '## BC 적용 가능성',transfer.to_csv(index=False),
        '현재 BC 6개월로 전년 대비 조건을 검증할 수 없습니다. 카페·주말·유동인구 조건은 측정할 수 없고 연령 코드는 코드북이 필요합니다. 이 결과만으로 새 전국 후보 순위를 만들지 않습니다.',
        '## 변수 정의',
        '모든 매출·건수·점포 지표는 음식업 CS100 코드 기준입니다. 유동인구는 업종과 무관한 행정동 추정 총량이며 실제 고유 방문객 수가 아닙니다. 20~30대 비중 분모는 연령별 매출 합계입니다. 개업·폐업률은 건수/현재 점포수로 계산한 분석용 비율이며 공식 제공 비율과 다를 수 있습니다. 음식업 매출 추정의 정확한 모집단·방법 변경은 공급기관 메타데이터 확인이 필요합니다.',
        '## 참고 근거',*[f'- [{title}]({url})' for title,url in SOURCES],
        '## 재현', '`uv run python src/hotplace_cases.py`', '원본은 읽기 전용 사용. 보고서와 표·그림은 reports/hotplace_cases에 저장합니다.']
    (OUT/'report.md').write_text('\n\n'.join(report),encoding='utf-8')
    print(summary.to_string(index=False))
    print(lead.to_string(index=False))
    print(match.to_string(index=False))


if __name__=='__main__':
    main()
