# 런던 공공자전거의 시간·대여소별 이용 패턴과 출발-도착 흐름 분석

LLM을 활용한 데이터 분석 (2026-2) 학기 프로젝트 | 혁신공유학부 배성윤 (2026-50713)

런던 공공자전거(TfL Santander Cycles)의 2023~2025년 대여 기록 약 2,600만 건을 분석해, **어느 대여소에서 언제 출발과 반납이 몰리는지**, 그 패턴이 **해마다 반복되는 구조인지**를 확인한다.

> 표기 원칙: **관찰**은 데이터에서 확인한 사실, **해석**은 그로부터의 추론이다. 외부 근거가 없는 해석은 "가설"로 표시한다.

---

## 진행 상태

| 단계 | 상태 | 제출물 |
|---|---|---|
| 1차 기획 | 완료 | [01_proposal/01_proposal.md](01_proposal/01_proposal.md) |
| 2차 데이터·전처리·EDA | 완료 | [02_data-eda/02_data-eda.md](02_data-eda/02_data-eda.md) |
| 분석·모델링·검증 | 완료 | [03_analysis-modeling/03_analysis-modeling.md](03_analysis-modeling/03_analysis-modeling.md) |
| 1차 피드백 추가 점검 | 완료 | [03_analysis-modeling/feedback_checks.md](03_analysis-modeling/feedback_checks.md) |
| 중간 보고서 (2차 발표 대체) | 11월 29일 제출 | [02_data-eda/02_midterm_report.pdf](02_data-eda/02_midterm_report.pdf) |
| 최종 보고서 (3차) | 12월 6일 제출 예정 | `04_final/` |

---

## 분석 질문

1. 근무일·비근무일의 시간대별 출발·도착 패턴으로 대여소를 몇 가지 유형으로 나눌 수 있으며, 그 유형은 해마다 유지되는가?
2. 시간대별로 출발과 도착이 크게 불균형한 대여소는 어디이며, 그 불균형은 하루 단위로 상쇄되는가, 해마다 반복되는가?
3. 대여 시작 시점 정보로 1회 대여 시간을 얼마나 설명·예측할 수 있는가? 전기자전거의 긴 대여 시간은 시간·장소 구성으로 설명되는가?

## 주요 결과 (현재까지)

| 질문 | 방법 | 결과 |
|---|---|---|
| 1. 대여소 유형 | 시간대별 출발·도착 비율 96개 + K-means (k = 3) | 대여소 796개가 **평일 오전 출발 집중형 429 / 시간대 분산형 228 / 평일 오전 도착 집중형 139**로 나뉨. 해마다 94~96%가 같은 유형 유지 |
| 2. 출발-도착 불균형 | 불균형 지수 I = (도착−출발)/(도착+출발), 하루 평균 순유입 | 평일 오전 첨두에는 대여소의 **48%**가 한쪽으로 3배 이상 치우치지만 **하루 전체로는 0개**. 오전·오후 순유입 상관 −0.93, 연도 간 순위 상관 0.982 |
| 3. 대여 시간 | log(대여 시간) 회귀, 2023~24 학습 / 2025 평가 | 최선 모델도 Baseline 대비 MAE **−3.9%** (R² 0.09). 전기자전거는 같은 조건에서도 **+1.6~2.3분** 더 김 |
| 범위 확인 | 2023년 8월만 따로 분석 | 시간대 모양 상관 0.992, 오전 불균형 순위 상관 0.972, 유형 일치 90.4% → 주요 결론 동일 |

자세한 수치와 검증은 [03_analysis-modeling.md](03_analysis-modeling/03_analysis-modeling.md)에 있다.

---

## 데이터

| 항목 | 내용 |
|---|---|
| 출처 | Transport for London(TfL) 공개 데이터, <https://cycling.data.tfl.gov.uk/#!usage-stats%2F> |
| 수집 방법 | usage-stats 폴더에서 2023~2025년을 포함하는 주 단위 CSV 86개를 직접 내려받음 (2026-11-14). 파일 목록: `02_data-eda/merge_manifest.csv` |
| 규모 | 원본 26,386,749행 (2022-12-26~2025-12-31) → 분석 26,126,975행 (2023-01-01~2025-12-31) |
| 단위 | 1행 = 대여 1건 (대여 번호, 시작·종료 시각, 출발·도착 대여소, 자전거 모델, 대여 시간) |
| 대여소 | 원본 ID 1,252개 → 이름 기준 규칙으로 통일 후 835개 |
| 외부 자료 | 잉글랜드·웨일스 공휴일 (GOV.UK) — `data/external/`에 스크립트가 생성 |
| 개인정보 | 이용자 정보 없음. 자전거 번호는 분석에 쓰지 않음 |

### 이용 조건

이 프로젝트는 Open Government Licence 2.0을 기반으로 한 TfL 공개 데이터 이용 조건에 따라 데이터를 사용한다.

**Powered by TfL Open Data.** 이 프로젝트는 TfL과 관련이 없으며 TfL의 보증을 받지 않았다.

원자료는 용량이 커서 저장소에 올리지 않는다. 재현하려면 위 주소에서 직접 내려받는다.

---

## 폴더 구조

```
data-analysis-project/
├─ README.md
├─ requirements.txt
├─ common.py                       공통 설정 (경로, 공휴일, 첨두 시간대, 그래프 스타일)
├─ 00_merge_raw.py ~ 08_feedback_checks.py
├─ 01_proposal/                    1차 기획서
├─ 02_data-eda/                    2차 문서, 중간 보고서, 통합·품질·전처리·EDA 보고서, 그래프
│  ├─ check_tables/                품질 점검·대여소 통일 상세표
│  └─ figures/
├─ 03_analysis-modeling/           분석 문서, 질문 1~3 보고서, 피드백 점검, 그래프
│  └─ figures/
├─ 04_final/                       최종 보고서 (예정)
├─ docs/
│  └─ preprocessing_decisions.md   전처리 결정 기록, 생성형 AI 검증 사례
├─ data/
│  ├─ raw/                         TfL 원자료 (저장소에 올리지 않음)
│  ├─ interim/                     형식 통일 parquet (올리지 않음)
│  ├─ processed/                   분석 데이터 (올리지 않음)
│  ├─ external/                    공휴일 목록
│  └─ agg/                         집계 테이블·결과표 (분석·앱에서 사용)
└─ app/                            Streamlit 앱 (예정)
```

---

## 실행 방법

### 1. 환경 준비

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. 데이터 준비

<https://cycling.data.tfl.gov.uk/#!usage-stats%2F>에서 2023~2025년 기간의 CSV를 내려받아 `data/raw/`에 넣는다. 하위 폴더도 읽는다.

### 3. 순서대로 실행 (프로젝트 폴더에서)

| 순서 | 스크립트 | 하는 일 | 주요 결과물 |
|---|---|---|---|
| 0 | `00_merge_raw.py` | 원자료 형식 통일 (값은 고치지 않음, 행 수 자동 검산) | `data/interim/trips/`, `02_data-eda/merge_report.md` |
| 1 | `01_check_data.py` | 결측·중복·이상값·자료형·대여소 ID 점검 | `02_data-eda/data_check_report.md` |
| 2 | `02_preprocess.py` | 행 제외 기준 적용, 대여소 통일, 파생 변수 | `data/processed/trips/`, `02_data-eda/preprocess_log.md` |
| 3 | `03_eda.py` | 탐색적 분석 집계와 그래프 | `data/agg/`, `02_data-eda/eda_report.md` |
| 4 | `04_build_analysis_tables.py` | 분석용 대여소×시간대 테이블, 질문 3 표본 | `03_analysis-modeling/tables_log.md` |
| 5 | `05_q1_station_types.py` | 질문 1: 대여소 유형 군집화와 검증 | `q1_report.md`, `data/agg/station_clusters.csv` |
| 6 | `06_q2_imbalance.py` | 질문 2: 시간대별 불균형 (5 결과 사용) | `q2_report.md`, `data/agg/station_imbalance.csv` |
| 7 | `07_q3_duration_model.py` | 질문 3: 대여 시간 모델 비교 (5 결과 사용) | `q3_report.md`, `q3_metrics.csv` |
| 8 | `08_feedback_checks.py` | 1차 피드백 추가 점검 (5 결과 사용) | `feedback_checks.md`, `data/agg/station_summary.csv` |

- 경로는 각 스크립트 파일 위치를 기준으로 계산하므로 프로젝트 폴더 바로 아래에 두고 실행한다.
- 각 단계는 읽은 행 수, 저장한 행 수, 집계 합계를 자동으로 검산해 보고서에 기록한다.
- 0~3단계는 원자료 전체를 읽으므로 시간이 걸린다. 4단계 이후는 `data/agg`와 `data/processed`를 사용한다.

---

## 1차 피드백 반영

| 피드백 | 반영 |
|---|---|
| 데이터 출처와 분석 범위 확인 | 데이터 개요표 작성, Kaggle 자료와 TfL 원자료 관계 확인(2023년 8월 행 수 일치), 3년 확장 근거 기록, 2023년 8월만 따로 분석해 결론 비교 |
| 군집 특징 설계 | 규모가 아닌 모양을 비교하는 시간대 비율 96개 특징, 표준화 여부 근거 기록, 피드백 제안 특징(표준화)으로 다시 군집화해 비교 |
| 관찰과 해석 구분 | 모든 결과를 관찰/해석으로 구분, 이용 목적은 가설로 표현, 군집 이름을 관찰된 특성으로 변경, OD는 경로가 아님을 명시 |

## 1차 이후 변경 사항

- **데이터**: Kaggle의 2023년 8월 자료(776,527행) → TfL 원자료 3년치(2023~2025년)로 확장
- **분석 질문**: 질문 1~3을 결과 검증이 가능한 형태로 다시 정리 (3차)
- **전처리**: 서머타임 처리, 대여소 ID 표기 형식(앞자리 0) 통일, 이름 변경 판단 규칙 추가
- **군집 이름**: 장소 추정("도심형" 등) → 관찰 특성("평일 오전 출발 집중형" 등)
- **일정**: 2차 발표는 중간 보고서로 대체, 최종 보고서는 3차(12월 6일)로 변경

## 생성형 AI 활용과 검증

코드 작성과 결과 정리에 생성형 AI를 활용했고, 실행 결과와 **정답을 아는 가짜 데이터**로 검증했다. 발견한 문제와 수정 20건을 기록했다.

- 기록: [docs/preprocessing_decisions.md](docs/preprocessing_decisions.md) (사례 1~12), [03_analysis-modeling.md](03_analysis-modeling/03_analysis-modeling.md) 9절 (사례 13~19)
- 대표 사례: AI가 대여소 ID 문제를 "ID 체계 변경"으로 해석 → 직접 세어 보니 앞자리 0 표기 차이(1,252 → 833) → "ID 표기 형식 변경"으로 정정

## 한계

- 대여소 위치 좌표가 없어 주변 환경과 이동 거리는 이름으로만 추정했다.
- 재고·용량·재배치 기록이 없어 불균형이 실제 자전거 부족으로 이어졌는지 판단할 수 없다.
- 이용자·날씨 정보가 없어 "통근", "여가"는 패턴에서 추론한 해석이다.
- 결과는 2023~2025년 런던 TfL 도크형 공공자전거에 한정된다.

## 참고 자료

- TfL Cycling Data, usage-stats: <https://cycling.data.tfl.gov.uk/#!usage-stats%2F>
- GOV.UK, UK bank holidays: <https://www.gov.uk/bank-holidays>
- GOV.UK, When do the clocks change: <https://www.gov.uk/when-do-the-clocks-change>
- Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2, 193–218.
