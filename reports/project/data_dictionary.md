# 프로젝트 데이터 사전

## 원천 데이터

| 원천 | 처리 기준 | 용도 |
|---|---|---|
| 서울시 추정매출-행정동 | 행정동·업종·분기 | 매출액, 매출건수, 미래 성장률 |
| 서울시 점포-행정동 | 행정동·업종·분기 | 점포수, 개업·폐업 관련 변수 |
| 서울시 길단위인구-행정동 | 행정동·분기 | 인구 규모 |
| 서울시 소비-행정동 | 행정동·분기 | 소비 규모 |
| mobile.xlsx | 시트별 원형 유지 | 모바일 인구 보조 분석 |

## 산출 데이터

- `merged_panel_data.csv`: 행정동 × 업종 × 분기 패널. 서울시 원천 테이블을 공통 키로 결합하고 시차·성장률·미래 목표를 포함합니다.
- `feature_data.csv`: 모델 입력 변수. `future_sales_growth`, `target_top20`, `source_file`은 제외했습니다.
- `target_data.csv`: `future_sales_growth`와 분기별 상위 20% 여부(`target_top20`).

## 주요 변수

| 변수 | 의미 |
|---|---|
| `quarter` | YYYYQ 형식의 분기 코드(예: 20241) |
| `admin_code` | 행정동 코드 |
| `service_code` | 서비스 업종 코드 |
| `sales_amount` | 추정 매출 금액 |
| `sales_count` | 추정 매출 건수 |
| `store_count` | 점포 수 |
| `population_total` | 행정동 인구 규모 |
| `consumption_total` | 행정동 소비 규모 |
| `sales_amount_lag1`, `sales_amount_lag2` | 1·2분기 전 매출 |
| `sales_amount_roll4` | 직전 최대 4개 분기 평균 매출 |
| `future_sales_growth` | 다음 분기 매출 성장률 |
| `target_top20` | 같은 분기 내 미래 성장률 상위 20% 여부 |

## 주의사항

- 현재 작업 폴더에는 BC카드 원천 데이터가 없어 BC 전국 단위 모델은 아직 만들지 않았습니다.
- 모바일 데이터는 행정동 코드와 직접 연결되는 키가 확인되지 않아 패널에 임의 결합하지 않았습니다.
- `target_top20`은 현재 확보된 분기 데이터로 정의한 대체 목표이며, 월별 BC카드 데이터가 추가되면 요구사항의 3개월 목표로 교체해야 합니다.
- `data/processed/preprocessed_data.csv`는 2026년 1~6월 BC카드 데이터로, 현재 학습·평가에서는 제외하고 향후 상권 후보 선정에만 사용합니다.
