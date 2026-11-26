# 전처리 결과

- 실행 일시: 2026-11-16 01:20
- pandas 3.1.1, pyarrow 26.0.1
- 기준의 근거와 대안: `docs/preprocessing_decisions.md`

## 1. 기준별 제외 건수 (적용 순서대로)

앞 기준에서 이미 빠진 행은 뒤 기준에서 다시 세지 않는다.

| 기준                       | 제외      | 전체 대비   |       남은 행 |
|:-------------------------|:--------|:--------|-----------:|
| 0. 통합 데이터                | -       | -       | 26,386,749 |
| 1. 분석 기간 밖               | 48,901  | 0.185%  | 26,337,848 |
| 2. 반납 정보 결측              | 184     | 0.001%  | 26,337,664 |
| 3. 종료<시작 (서머타임으로 설명 안 됨) | 0       | 0.000%  | 26,337,664 |
| 4. 내용 중복 (사본)            | 237     | 0.001%  | 26,337,427 |
| 5. 짧은 왕복 (같은 대여소, 1분 이하) | 189,287 | 0.717%  | 26,148,140 |
| 6. 긴 대여 (24시간 초과)        | 21,165  | 0.080%  | 26,126,975 |

- 최종 분석 데이터: **26,126,975행** (통합 데이터의 99.016%)
- 검산: 통합 - 제외 합계 = 26,126,975 → 일치

## 2. 종료 < 시작 행의 서머타임 확인

- 서머타임 해제일(10월 마지막 일요일): 2023-10-29, 2024-10-27, 2025-10-26, 2026-10-25
- 분석 기간 안의 종료 < 시작 행: 185건
  - 해제일에 발생하고 시각 차이 60분 이내 → 유지: 185건
  - 그 밖 → 제외: 0건

발생 날짜 (상위 10):

| 시작 날짜      |   건수 |
|:-----------|-----:|
| 2024-10-27 |   71 |
| 2025-10-26 |   66 |
| 2023-10-29 |   48 |

## 3. 대여소 통일

- 원본 (ID, 이름) 쌍: 1,268개, 원본 ID: 1,252개, 원본 이름: 845개
- 통일 후 대여소(station_key): 835개
- 앞자리 0을 무시하면 원본 ID 수: 833개 (같은 번호가 0을 붙인 형태와 안 붙인 형태로 모두 나타나는 경우: 419개, 예: `000964`/`964`)

규칙별 적용 현황 (대여건수는 출발·도착을 각각 세어 실제 대여 건수의 약 2배):

| rule         |    쌍 |   대여건수 (출발+도착 합산) |
|:-------------|-----:|------------------:|
| A_표기정리       |   77 |           2999465 |
| B_표기변형통일     |    5 |             25474 |
| D_이름변경통합     |    5 |            114903 |
| 같은ID_다른이름_유지 |    5 |             27678 |
| 변경없음         | 1176 |          49605794 |

- ID 2개 이상이 합쳐진 대여소: 422개
  - 그중 앞자리 0 표기 차이만으로 설명되는 대여소(예: `000964`/`964`): 419개
  - 번호 자체가 다른 ID가 합쳐진 대여소: 3개 ['Coomer Place, West Kensington', 'Fore Street, Guildhall', 'Fore Street Avenue, Guildhall']
  - 그중 여러 ID가 함께 쓰인 달이 3개월 이상인 대여소: 0개 (한 달 30건 이상 쓰인 경우만 '사용 중'으로 셈)
    → 여러 ID가 오래 동시에 쓰인 대여소가 없으므로, 같은 대여소의 ID 표기가 시기별로 바뀌었다는 해석과 맞는다.

함께 쓰인 달 수의 분포 (ID 2개 이상인 대여소):

|   함께 쓰인 달 수 |   대여소 수 |
|------------:|--------:|
|           0 |     419 |
|           1 |       3 |

규칙 B: 표기 변형으로 통일된 이름 (전체, 검토 필요):

|   raw_id | raw_name                           | station_key                         |   similarity_to_dominant |   trips |
|---------:|:-----------------------------------|:------------------------------------|-------------------------:|--------:|
|   000964 | Bath Street, St. Lke's             | Bath Street, St. Luke's             |                    0.978 |   10247 |
|      964 | Bath Street, St. Lke's             | Bath Street, St. Luke's             |                    0.978 |    2876 |
|   001212 | Concert Hall Approach, South Bank  | Concert Hall Approach 2, South Bank |                    0.971 |    6435 |
|   300100 | Limburg Road, Clapham Junction_OLD | Limburg Road, Clapham Junction      |                    0.938 |    1710 |
|   300204 | Victoria Rise, Clapham Common_OLD  | Victoria Rise, Clapham Common       |                    0.935 |    4206 |

규칙 D: 같은 ID의 다른 이름 판정 (전체):

|   raw_id | main_name                           | other_name                     |   concurrent_months |   other_trips | decision                       |
|---------:|:------------------------------------|:-------------------------------|--------------------:|--------------:|:-------------------------------|
|   300006 | Clapham South, Clapham South        | Coomer Place, West Kensington  |                   0 |         11094 | 분리 유지 (다른 지역)                  |
|   300099 | Abyssinia Close, Clapham Junction   | Adeline Place, Bloomsbury      |                   0 |          1706 | 분리 유지 (다른 지역)                  |
|   002635 | Concert Hall Approach 1, South Bank | Agar Grove, Camden Town        |                   0 |            68 | 분리 유지 (다른 지역)                  |
|   300222 | Fore Street, Guildhall              | Fore Street Avenue, Guildhall  |                   1 |          7761 | 분리 유지 (운영 중인 다른 ID 200253의 이름) |
|   200253 | Fore Street Avenue, Guildhall       | Fore Street, Guildhall         |                   1 |          7049 | 분리 유지 (운영 중인 다른 ID 300222의 이름) |
|   001011 | Argyle Street, Kings Cross          | Belgrove Street, King's Cross  |                   1 |         36691 | 통합 (이름 변경)                     |
|   001006 | Bayley Street, Bloomsbury           | Percy Street, Bloomsbury       |                   0 |         32270 | 통합 (이름 변경)                     |
|   001145 | Earnshaw Street, Covent Garden      | Bucknall Street, Covent Garden |                   0 |         31859 | 통합 (이름 변경)                     |
|   200118 | Parkway, Camden Town                | Albert Street, Camden Town     |                   0 |         14082 | 통합 (이름 변경)                     |
|   200025 | London Fields, Hackney Central      | London Fields, Hackney         |                   0 |             1 | 통합 (이름 변경)                     |

판정 요약:

| 판정                             |   건수 |
|:-------------------------------|-----:|
| 통합 (이름 변경)                     |    5 |
| 분리 유지 (다른 지역)                  |    3 |
| 분리 유지 (운영 중인 다른 ID 200253의 이름) |    1 |
| 분리 유지 (운영 중인 다른 ID 300222의 이름) |    1 |

이름 변경으로 통합된 대여소:

| station_key                    | 합쳐진_이름                                                          | ID            |   대여건수 |
|:-------------------------------|:----------------------------------------------------------------|:--------------|-------:|
| Argyle Street, Kings Cross     | Argyle Street, Kings Cross / Belgrove Street, King's Cross      | 001011 / 1011 | 223306 |
| Bayley Street, Bloomsbury      | Bayley Street, Bloomsbury / Percy Street, Bloomsbury            | 001006 / 1006 |  81270 |
| Earnshaw Street, Covent Garden | Bucknall Street, Covent Garden / Earnshaw Street, Covent Garden | 001145 / 1145 |  92939 |
| London Fields, Hackney Central | London Fields, Hackney / London Fields, Hackney Central         | 200025        | 130805 |
| Parkway, Camden Town           | Albert Street, Camden Town / Parkway, Camden Town               | 200118        |  38607 |

- 상세: `02_data-eda/check_tables/station_mapping.csv`, `station_key_summary.csv`, `station_rename_checks.csv`

