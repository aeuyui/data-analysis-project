"""
2차 2단계: 전처리 스크립트

목적
- 01_check_data.py의 점검 결과를 근거로 정한 전처리 기준을 적용해 분석용 데이터를 만든다.
- 기준별 제외 건수를 '적용 순서대로' 기록해, 어떤 기준으로 몇 건이 빠졌는지 설명할 수 있게 한다.
- 원본(data/raw)과 통합 데이터(data/interim)는 수정하지 않는다.

전처리 기준 (근거와 대안은 docs/preprocessing_decisions.md 참고)
  1. 분석 기간: 2023-01-01 ~ 2025-12-31 (대여 시작 기준). 2022-12는 26일부터만 있어 제외
  2. 반납 정보 결측(종료 일시·도착 대여소·대여 시간 중 하나라도 결측) 제외
  3. 종료 < 시작: 서머타임 해제일(10월 마지막 일요일)로 설명되는 건은 유지, 나머지 제외
  4. 내용 중복(같은 자전거·시작·종료·출발·도착): 묶음마다 trip_id가 가장 작은 1건만 남김
  5. 짧은 왕복: 출발 대여소 = 도착 대여소(통일된 대여소 기준) 이면서 1분 이하 → 제외
  6. 긴 대여: 24시간 초과 → 제외
  * 대여 시간은 duration_ms를 사용한다 (시작·종료 시각은 현지 시계 시각이라 서머타임 전환일에 1시간 차이 발생)

대여소 통일 (분석 단위 = station_key)
  A. 이름 표기 정리: 앞뒤 공백 제거, 연속 공백 1개로, 쉼표 앞 공백 제거, 쉼표 뒤 공백 1개
  B. 같은 ID 안의 표기 변형 통일: 같은 ID에서 가장 많이 쓰인 이름과 유사도 0.9 이상이면 그 이름으로 통일
     (예: 'St. Lke's' → 'St. Luke's'). 유사도가 낮은 다른 이름은 다른 대여소로 유지(위치 이전 가능성, 좌표 없어 확인 불가)
  C. 정리된 이름을 station_key로 사용 → ID 체계가 바뀌어도 같은 이름이면 같은 대여소로 본다
     단, 같은 이름의 여러 ID가 여러 달 동안 '동시에' 쓰였다면 실제로 다른 대여소일 수 있어 검토 대상으로 표시한다.
  D. 같은 ID의 다른 이름이 '같은 지역(쉼표 뒤 지역명)'이고 동시에 쓰인 달이 3개월 미만이면 이름 변경으로 보고 통합
     (예: 'Bayley Street, Bloomsbury' → 'Percy Street, Bloomsbury'). 지역이 다르면 ID 재사용으로 보고 분리 유지.
     단, 그 이름이 같은 기간에 운영 중인 '다른 ID'의 대표 이름이면 별개 대여소로 보고 통합하지 않는다
     (예: Fore Street(300222)와 Fore Street Avenue(200253)는 3년 내내 함께 운영됨 → 두 대여소).
     통합된 이름 묶음은 가장 많이 쓰인 이름을 station_key로 쓴다.
  → 결과 대응표는 02_data-eda/check_tables/station_mapping.csv 에서 반드시 검토한다.

파생 변수
  duration_min, year, month, season, start_date, start_hour, day_of_week(0=월), is_weekend,
  start_station_key, end_station_key, is_round_trip

폴더 구조 (이 스크립트는 proj 폴더 바로 아래에 둔다)
    proj/
    ├─ 00_merge_raw.py
    ├─ 01_check_data.py
    ├─ 02_preprocess.py          ← 이 파일
    ├─ data/interim/trips/       ← 입력
    ├─ data/processed/trips/     ← 출력 (분석용 데이터)
    └─ 02_data-eda/              ← 처리 기록 저장

사용법 (proj 폴더에서 실행)
    python 02_preprocess.py

결과
- data/processed/trips/<원본파일명>.parquet
- 02_data-eda/preprocess_log.md                       : 기준별 제외 건수, 서머타임 확인, 대여소 통일 요약
- 02_data-eda/check_tables/station_mapping.csv        : (원본 ID, 원본 이름) → station_key 대응표
- 02_data-eda/check_tables/station_key_summary.csv    : station_key별 ID 목록과 사용 기간
- 02_data-eda/check_tables/station_rename_checks.csv  : 같은 ID의 다른 이름 판정 결과 (규칙 D)
- 02_data-eda/check_tables/end_before_start_rows.csv  : 종료 < 시작 행과 서머타임 판정 결과
"""

import gc
import re
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# 경로 (이 스크립트가 있는 폴더 기준)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
IN_DIR = PROJECT_DIR / "data" / "interim" / "trips"
OUT_DIR = PROJECT_DIR / "data" / "processed" / "trips"
REPORT_DIR = PROJECT_DIR / "02_data-eda"
TABLE_DIR = REPORT_DIR / "check_tables"
LOG_PATH = REPORT_DIR / "preprocess_log.md"

# ---------------------------------------------------------------------------
# 전처리 기준값 (바꾸면 결과와 로그가 함께 바뀐다)
# ---------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2023-01-01")
PERIOD_END = pd.Timestamp("2026-01-01")          # 미포함
SHORT_ROUND_TRIP_MS = 60_000                     # 1분 이하 왕복 제외
LONG_TRIP_MS = 24 * 60 * 60 * 1000               # 24시간 초과 제외
NAME_SIMILARITY = 0.9                            # 같은 ID 안의 표기 변형 판단 기준
OVERLAP_MIN_TRIPS = 30                           # 한 달에 이 건수 이상 쓰인 ID만 '사용 중'으로 본다
OVERLAP_REVIEW_MONTHS = 3                        # 여러 ID가 함께 쓰인 달이 이 이상이면 검토 대상으로 표시

DICT_COLUMNS = ["start_station_id", "start_station_name", "end_station_id", "end_station_name",
                "bike_id", "bike_model", "duration_source", "source_file"]
DUP_KEY = ["bike_id", "start_time", "end_time", "start_station_id", "end_station_id"]
SEP = "\x1f"  # (ID, 이름) 쌍을 하나의 문자열 키로 만들 때 쓰는 구분자


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------
def to_markdown(df: pd.DataFrame, index=False) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


def last_sunday(year: int, month: int) -> date:
    """해당 월의 마지막 일요일 (영국 서머타임 시작: 3월, 해제: 10월)."""
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - 6) % 7)


def normalize_name(name) -> str:
    """대여소 이름 표기 정리 (규칙 A)."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return name
    s = " ".join(str(name).split())
    s = re.sub(r"\s*,\s*", ", ", s)
    return s


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def area_of(name: str) -> str:
    """이름에서 지역명(마지막 쉼표 뒤)을 비교용으로 꺼낸다. 예: "Bath Street, St. Luke's" → "st lukes"."""
    if not isinstance(name, str) or "," not in name:
        return ""
    area = re.sub(r"[^a-z0-9 ]", "", name.rsplit(",", 1)[1].lower())
    return " ".join(area.split())


def areas_match(a: str, b: str) -> bool:
    """지역명이 같거나 한쪽이 다른 쪽을 포함하면(예: Hackney / Hackney Central) 같은 지역으로 본다."""
    x, y = area_of(a), area_of(b)
    if not x or not y:
        return False
    return x == y or x.startswith(y + " ") or y.startswith(x + " ")


def pct(part, whole) -> str:
    return f"{part / whole * 100:.3f}%" if whole else "-"


def season_of(month: pd.Series) -> pd.Series:
    """기상학적 계절 (영국 Met Office 등에서 쓰는 3개월 단위 구분)."""
    m = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
         6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}
    return month.map(m).astype("category")


# ---------------------------------------------------------------------------
# 1단계(전체): 대여소 대응표 만들기
# ---------------------------------------------------------------------------
def build_station_mapping():
    ds = pq.ParquetDataset(str(IN_DIR), read_dictionary=DICT_COLUMNS)
    parts, month_parts = [], []
    for side in ["start", "end"]:
        df = ds.read(columns=[f"{side}_station_id", f"{side}_station_name", "start_time"]).to_pandas()
        keys = [f"{side}_station_id", f"{side}_station_name"]
        g = df.groupby(keys, observed=True)["start_time"]
        agg = g.agg(trips="size", first_seen="min", last_seen="max").reset_index()
        agg.columns = ["raw_id", "raw_name", "trips", "first_seen", "last_seen"]
        parts.append(agg)
        # 월별 사용 건수 (ID 교체 시기 판단용)
        df["ym"] = (df["start_time"].dt.year * 100 + df["start_time"].dt.month).astype("int32")
        mon = df.groupby(keys + ["ym"], observed=True).size().reset_index(name="trips")
        mon.columns = ["raw_id", "raw_name", "ym", "trips"]
        month_parts.append(mon)
        del df, g, mon
        gc.collect()

    pairs = (pd.concat(parts)
             .astype({"raw_id": "string", "raw_name": "string"})
             .groupby(["raw_id", "raw_name"])
             .agg(trips=("trips", "sum"), first_seen=("first_seen", "min"), last_seen=("last_seen", "max"))
             .reset_index())

    # 규칙 A: 표기 정리
    pairs["name_norm"] = pairs["raw_name"].map(normalize_name)
    pairs["rule"] = (pairs["name_norm"] != pairs["raw_name"]).map({True: "A_표기정리", False: "변경없음"})

    # 규칙 B: 같은 ID 안의 표기 변형 → 가장 많이 쓰인 이름으로
    by_id_name = pairs.groupby(["raw_id", "name_norm"], as_index=False)["trips"].sum()
    dominant = (by_id_name.sort_values(["raw_id", "trips"], ascending=[True, False])
                .drop_duplicates("raw_id").set_index("raw_id")["name_norm"])
    pairs["dominant_name"] = pairs["raw_id"].map(dominant)
    pairs["similarity_to_dominant"] = [
        round(similarity(n, d), 3) for n, d in zip(pairs["name_norm"], pairs["dominant_name"])
    ]
    is_variant = (pairs["name_norm"] != pairs["dominant_name"]) & (pairs["similarity_to_dominant"] >= NAME_SIMILARITY)
    is_other = (pairs["name_norm"] != pairs["dominant_name"]) & ~is_variant
    pairs["station_key"] = pairs["name_norm"].where(~is_variant, pairs["dominant_name"])
    pairs["name_v"] = pairs["station_key"]  # 규칙 A·B 적용 후 이름

    # 월별 사용 건수 (규칙 D의 '동시 사용' 판단과 station_key 요약에 사용)
    months = (pd.concat(month_parts).astype({"raw_id": "string", "raw_name": "string"})
              .groupby(["raw_id", "raw_name", "ym"], as_index=False)["trips"].sum())
    pair_index = pd.MultiIndex.from_frame(months[["raw_id", "raw_name"]])
    months["name_v"] = pairs.set_index(["raw_id", "raw_name"])["name_v"].reindex(pair_index).values

    # 규칙 D: 같은 ID의 다른 이름이 '이름 변경'인지 판단해 같은 대여소로 통합
    #   통합 조건: 지역명(쉼표 뒤)이 같고, 두 이름이 같은 달에 함께 쓰인 기간이 짧음
    #   - 지역이 다르면 ID 재사용(다른 대여소)으로 보고 분리 유지
    #   - 같은 지역이어도 여러 달 동시에 쓰였다면 별개 대여소일 수 있어 분리 유지
    active = months.groupby(["raw_id", "name_v", "ym"], as_index=False)["trips"].sum()
    active = active[active["trips"] >= OVERLAP_MIN_TRIPS]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # 각 ID의 대표 이름과 '사용 중'인 달 (다른 대여소의 이름인지 확인용)
    id_main_name = (pairs.groupby(["raw_id", "name_v"], as_index=False)["trips"].sum()
                    .sort_values(["raw_id", "trips"], ascending=[True, False])
                    .drop_duplicates("raw_id").set_index("raw_id")["name_v"])
    id_active = (months.groupby(["raw_id", "ym"])["trips"].sum().reset_index()
                 .query("trips >= @OVERLAP_MIN_TRIPS").groupby("raw_id")["ym"].agg(set))
    owners_of = id_main_name.reset_index().groupby("name_v")["raw_id"].agg(list)

    checks = []
    for rid, grp in pairs.groupby("raw_id"):
        names = grp.groupby("name_v")["trips"].sum().sort_values(ascending=False)
        if len(names) < 2:
            continue
        main = names.index[0]
        act = active[active["raw_id"] == rid]
        main_months = set(act.loc[act["name_v"] == main, "ym"])
        my_active = id_active.get(rid, set())
        for other in names.index[1:]:
            same_area = areas_match(main, other)
            concurrent = len(main_months & set(act.loc[act["name_v"] == other, "ym"]))
            # 'other'가 같은 기간에 운영 중인 다른 ID의 대표 이름이면 → 별개 대여소의 이름 (예: 이름이 서로 뒤바뀌어 기록된 경우)
            rival = [r for r in owners_of.get(other, []) if r != rid
                     and len(my_active & id_active.get(r, set())) >= OVERLAP_REVIEW_MONTHS]
            if not same_area:
                decision = "분리 유지 (다른 지역)"
            elif concurrent >= OVERLAP_REVIEW_MONTHS:
                decision = "분리 유지 (동시 사용)"
            elif rival:
                decision = f"분리 유지 (운영 중인 다른 ID {rival[0]}의 이름)"
            else:
                decision = "통합 (이름 변경)"
                parent[find(other)] = find(main)
            checks.append({"raw_id": rid, "main_name": main, "other_name": other,
                           "main_area": area_of(main), "other_area": area_of(other),
                           "concurrent_months": concurrent, "other_trips": int(names[other]),
                           "decision": decision})
    rename_checks = pd.DataFrame(checks, columns=["raw_id", "main_name", "other_name", "main_area", "other_area",
                                                  "concurrent_months", "other_trips", "decision"])

    # 통합된 이름 묶음마다 가장 많이 쓰인 이름을 station_key로
    name_trips = pairs.groupby("name_v")["trips"].sum().sort_values(ascending=False)
    root_of = {n: find(n) for n in name_trips.index}
    rep = {}
    for n in name_trips.index:  # 건수 많은 순서이므로 처음 만나는 이름이 대표
        rep.setdefault(root_of[n], n)
    pairs["station_key"] = pairs["name_v"].map(lambda n: rep[root_of[n]])

    # 규칙 라벨 (뒤에 오는 것이 우선)
    id_main_key = (pairs.groupby(["raw_id", "station_key"], as_index=False)["trips"].sum()
                   .sort_values(["raw_id", "trips"], ascending=[True, False])
                   .drop_duplicates("raw_id").set_index("raw_id")["station_key"])
    pairs["id_main_key"] = pairs["raw_id"].map(id_main_key)
    pairs["rule"] = "변경없음"
    pairs.loc[pairs["name_norm"] != pairs["raw_name"], "rule"] = "A_표기정리"
    pairs.loc[pairs["station_key"] != pairs["name_v"], "rule"] = "D_이름변경통합"
    pairs.loc[pairs["name_v"] != pairs["name_norm"], "rule"] = "B_표기변형통일"
    pairs.loc[pairs["station_key"] != pairs["id_main_key"], "rule"] = "같은ID_다른이름_유지"

    # station_key 요약: 몇 개의 ID가 합쳐졌는지, ID들이 같은 달에 함께 쓰였는지
    #   - ID 체계 변경이라면 옛 ID가 끝날 무렵 새 ID가 시작되어, 함께 쓰인 달이 거의 없다.
    #   - 실제로 다른 대여소(같은 이름)라면 여러 달에 걸쳐 함께 쓰인다.
    #   - 최초·최종 사용일만 보면 기록 몇 건으로도 겹쳐 보이므로, 월별로 '대여 건수 기준'을 함께 본다.
    key_of = pairs.set_index(["raw_id", "raw_name"])["station_key"]
    months["station_key"] = key_of.reindex(pair_index).values
    id_month = months.groupby(["station_key", "raw_id", "ym"], as_index=False)["trips"].sum()
    id_month = id_month[id_month["trips"] >= OVERLAP_MIN_TRIPS]  # 월 몇 건 미만은 잔여 기록으로 보고 제외

    summaries = []
    for key, grp in pairs.groupby("station_key"):
        ids = grp.groupby("raw_id").agg(first_seen=("first_seen", "min"), last_seen=("last_seen", "max"),
                                        trips=("trips", "sum")).sort_values("first_seen")
        im = id_month[id_month["station_key"] == key]
        shared_months = int((im.groupby("ym")["raw_id"].nunique() >= 2).sum()) if len(ids) > 1 else 0
        summaries.append({
            "station_key": key,
            "n_ids": len(ids),
            "ids": " | ".join(ids.index),
            "id_periods": " | ".join(f"{i}:{r.first_seen:%Y-%m-%d}~{r.last_seen:%Y-%m-%d}" for i, r in ids.iterrows()),
            "shared_months": shared_months,
            "review_needed": shared_months >= OVERLAP_REVIEW_MONTHS,
            "trips": int(ids["trips"].sum()),
        })
    summary = pd.DataFrame(summaries).sort_values("trips", ascending=False)

    pairs = pairs.sort_values(["station_key", "raw_id", "trips"], ascending=[True, True, False])
    mapping = dict(zip(pairs["raw_id"] + SEP + pairs["raw_name"], pairs["station_key"]))
    return mapping, pairs, summary, rename_checks


# ---------------------------------------------------------------------------
# 1단계(전체): 제외할 내용 중복 trip_id 찾기
# ---------------------------------------------------------------------------
def find_duplicate_trip_ids() -> set:
    ds = pq.ParquetDataset(str(IN_DIR), read_dictionary=DICT_COLUMNS)
    df = ds.read(columns=["trip_id"] + DUP_KEY).to_pandas()
    df = df.sort_values("trip_id")
    drop = set(df.loc[df.duplicated(subset=DUP_KEY, keep="first"), "trip_id"].tolist())
    del df
    gc.collect()
    return drop


# ---------------------------------------------------------------------------
# 2단계(파일별): 기준 적용과 파생 변수
# ---------------------------------------------------------------------------
def dst_fallback_dates(years) -> set:
    return {last_sunday(y, 10) for y in years}


def process_file(path: Path, mapping: dict, drop_ids: set, fallback: set, counts: dict, ebs_rows: list):
    df = pq.read_table(path).to_pandas()
    counts["0. 통합 데이터"] += len(df)

    # 1. 분석 기간
    keep = (df["start_time"] >= PERIOD_START) & (df["start_time"] < PERIOD_END)
    counts["1. 분석 기간 밖"] += int((~keep).sum())
    df = df[keep]

    # 2. 반납 정보 결측
    missing = df["end_time"].isna() | df["end_station_id"].isna() | df["end_station_name"].isna() | df["duration_ms"].isna()
    counts["2. 반납 정보 결측"] += int(missing.sum())
    df = df[~missing]

    # 3. 종료 < 시작 (서머타임 해제로 설명되는지 확인)
    ebs = df["end_time"] < df["start_time"]
    if ebs.any():
        sub = df.loc[ebs, ["trip_id", "start_time", "end_time", "duration_ms", "source_file"]].copy()
        gap_min = (sub["start_time"] - sub["end_time"]).dt.total_seconds() / 60
        on_fallback = sub["start_time"].dt.date.isin(fallback) | sub["end_time"].dt.date.isin(fallback)
        sub["dst_explained"] = on_fallback & (gap_min <= 60) & (sub["duration_ms"] >= 0)
        ebs_rows.append(sub)
        bad_ids = set(sub.loc[~sub["dst_explained"], "trip_id"])
        counts["3. 종료<시작 (서머타임으로 설명 안 됨)"] += len(bad_ids)
        df = df[~df["trip_id"].isin(bad_ids)]

    # 4. 내용 중복
    dup = df["trip_id"].isin(drop_ids)
    counts["4. 내용 중복 (사본)"] += int(dup.sum())
    df = df[~dup]

    # 대여소 통일 (짧은 왕복 판단에 필요하므로 여기서 계산)
    for side in ["start", "end"]:
        k = df[f"{side}_station_id"].astype("string") + SEP + df[f"{side}_station_name"].astype("string")
        df[f"{side}_station_key"] = k.map(mapping).astype("string")
    unmapped = int(df["start_station_key"].isna().sum() + df["end_station_key"].isna().sum())
    if unmapped:
        raise RuntimeError(f"{path.name}: 대응표에 없는 대여소가 {unmapped}건 있습니다.")
    df["is_round_trip"] = df["start_station_key"] == df["end_station_key"]

    # 5. 짧은 왕복
    short = df["is_round_trip"] & (df["duration_ms"] <= SHORT_ROUND_TRIP_MS)
    counts["5. 짧은 왕복 (같은 대여소, 1분 이하)"] += int(short.sum())
    df = df[~short]

    # 6. 긴 대여
    long_ = df["duration_ms"] > LONG_TRIP_MS
    counts["6. 긴 대여 (24시간 초과)"] += int(long_.sum())
    df = df[~long_]

    # 파생 변수
    st = df["start_time"]
    df["duration_min"] = df["duration_ms"] / 60000
    df["year"] = st.dt.year.astype("int16")
    df["month"] = st.dt.month.astype("int8")
    df["season"] = season_of(df["month"])
    df["start_date"] = st.dt.normalize()
    df["start_hour"] = st.dt.hour.astype("int8")
    df["day_of_week"] = st.dt.dayofweek.astype("int8")  # 0=월요일
    df["is_weekend"] = df["day_of_week"] >= 5

    cols = ["trip_id", "start_time", "end_time", "duration_ms", "duration_min",
            "start_station_key", "end_station_key", "is_round_trip",
            "start_station_id", "start_station_name", "end_station_id", "end_station_name",
            "bike_id", "bike_model", "year", "month", "season", "start_date", "start_hour",
            "day_of_week", "is_weekend", "source_file"]
    df = df[cols]
    counts["최종 분석 데이터"] += len(df)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), OUT_DIR / path.name, compression="zstd")


# ---------------------------------------------------------------------------
# 기록
# ---------------------------------------------------------------------------
def write_log(counts, pairs, summary, rename_checks, ebs_all, fallback):
    total = counts["0. 통합 데이터"]
    lines = ["# 전처리 결과", "",
             f"- 실행 일시: {datetime.now():%Y-%m-%d %H:%M}",
             f"- pandas {pd.__version__}, pyarrow {pa.__version__}",
             "- 기준의 근거와 대안: `docs/preprocessing_decisions.md`", ""]

    lines += ["## 1. 기준별 제외 건수 (적용 순서대로)", "",
              "앞 기준에서 이미 빠진 행은 뒤 기준에서 다시 세지 않는다.", ""]
    rows, remain = [], total
    for k, v in counts.items():
        if k in ("0. 통합 데이터", "최종 분석 데이터"):
            continue
        remain -= v
        rows.append({"기준": k, "제외": f"{v:,}", "전체 대비": pct(v, total), "남은 행": f"{remain:,}"})
    table = pd.DataFrame([{"기준": "0. 통합 데이터", "제외": "-", "전체 대비": "-", "남은 행": f"{total:,}"}] + rows)
    lines += [to_markdown(table), ""]
    final = counts["최종 분석 데이터"]
    lines.append(f"- 최종 분석 데이터: **{final:,}행** (통합 데이터의 {pct(final, total)})")
    lines.append(f"- 검산: 통합 - 제외 합계 = {total - sum(v for k, v in counts.items() if k[0].isdigit() and k[0] != '0'):,}"
                 f" → {'일치' if remain == final else '불일치 (확인 필요)'}")
    lines.append("")

    lines += ["## 2. 종료 < 시작 행의 서머타임 확인", "",
              f"- 서머타임 해제일(10월 마지막 일요일): {', '.join(str(d) for d in sorted(fallback))}"]
    if ebs_all is not None and len(ebs_all):
        explained = int(ebs_all["dst_explained"].sum())
        lines.append(f"- 분석 기간 안의 종료 < 시작 행: {len(ebs_all):,}건")
        lines.append(f"  - 해제일에 발생하고 시각 차이 60분 이내 → 유지: {explained:,}건")
        lines.append(f"  - 그 밖 → 제외: {len(ebs_all) - explained:,}건")
        by_date = ebs_all["start_time"].dt.date.value_counts().head(10).rename_axis("시작 날짜").to_frame("건수").reset_index()
        lines += ["", "발생 날짜 (상위 10):", "", to_markdown(by_date)]
    else:
        lines.append("- 분석 기간 안에 종료 < 시작 행 없음")
    lines.append("")

    lines += ["## 3. 대여소 통일", ""]
    lines.append(f"- 원본 (ID, 이름) 쌍: {len(pairs):,}개, 원본 ID: {pairs['raw_id'].nunique():,}개, "
                 f"원본 이름: {pairs['raw_name'].nunique():,}개")
    lines.append(f"- 통일 후 대여소(station_key): {pairs['station_key'].nunique():,}개")

    # ID 표기 형식 차이 확인 (예: 000964 / 964)
    ids = pd.Series(pairs["raw_id"].unique(), dtype="string")
    stripped = ids.str.lstrip("0")
    padded_groups = int((stripped.value_counts() > 1).sum())
    lines.append(f"- 앞자리 0을 무시하면 원본 ID 수: {stripped.nunique():,}개 "
                 f"(같은 번호가 0을 붙인 형태와 안 붙인 형태로 모두 나타나는 경우: {padded_groups:,}개, 예: `000964`/`964`)")

    rule_counts = pairs.groupby("rule").agg(쌍=("raw_id", "size"), 대여건수=("trips", "sum")).reset_index()
    rule_counts = rule_counts.rename(columns={"대여건수": "대여건수 (출발+도착 합산)"})
    lines += ["", "규칙별 적용 현황 (대여건수는 출발·도착을 각각 세어 실제 대여 건수의 약 2배):", "",
              to_markdown(rule_counts), ""]

    multi = summary[summary["n_ids"] > 1]
    n_review = int(multi["review_needed"].sum())
    pad_only = multi["ids"].map(lambda s: len({i.lstrip("0") for i in s.split(" | ")}) == 1).astype(bool)
    lines.append(f"- ID 2개 이상이 합쳐진 대여소: {len(multi):,}개")
    lines.append(f"  - 그중 앞자리 0 표기 차이만으로 설명되는 대여소(예: `000964`/`964`): {int(pad_only.sum()):,}개")
    lines.append(f"  - 번호 자체가 다른 ID가 합쳐진 대여소: {int((~pad_only).sum()):,}개 "
                 f"{list(multi.loc[~pad_only, 'station_key'].head(10))}")
    lines.append(f"  - 그중 여러 ID가 함께 쓰인 달이 {OVERLAP_REVIEW_MONTHS}개월 이상인 대여소: {n_review:,}개 "
                 f"(한 달 {OVERLAP_MIN_TRIPS}건 이상 쓰인 경우만 '사용 중'으로 셈)")
    if n_review:
        lines.append("    → ID 체계 변경이라면 함께 쓰인 달이 거의 없어야 하므로, 이 대여소들은 실제로 다른 대여소일 수 있어 우선 검토")
        review = multi[multi["review_needed"]][["station_key", "ids", "shared_months", "trips"]]
        lines += ["", "검토 대상 (상위 30):", "", to_markdown(review.head(30))]
    else:
        lines.append("    → 여러 ID가 오래 동시에 쓰인 대여소가 없으므로, 같은 대여소의 ID 표기가 시기별로 바뀌었다는 해석과 맞는다.")
    lines.append("")
    lines.append("함께 쓰인 달 수의 분포 (ID 2개 이상인 대여소):")
    lines += ["", to_markdown(multi["shared_months"].value_counts().sort_index()
                              .rename_axis("함께 쓰인 달 수").to_frame("대여소 수").reset_index())]

    variants = pairs[pairs["rule"] == "B_표기변형통일"][["raw_id", "raw_name", "station_key", "similarity_to_dominant", "trips"]]
    if len(variants):
        lines += ["", "규칙 B: 표기 변형으로 통일된 이름 (전체, 검토 필요):", "", to_markdown(variants.head(50))]
        if len(variants) > 50:
            lines.append(f"- … 외 {len(variants) - 50}건 (station_mapping.csv 참고)")

    if len(rename_checks):
        rc = rename_checks.sort_values(["decision", "other_trips"], ascending=[True, False])
        lines += ["", "규칙 D: 같은 ID의 다른 이름 판정 (전체):", "",
                  to_markdown(rc[["raw_id", "main_name", "other_name", "concurrent_months", "other_trips", "decision"]])]
        lines.append("")
        lines.append("판정 요약:")
        lines += ["", to_markdown(rc["decision"].value_counts().rename_axis("판정").to_frame("건수").reset_index())]

    merged = pairs[pairs["rule"] == "D_이름변경통합"]
    if len(merged):
        groups = (pairs[pairs["station_key"].isin(merged["station_key"])]
                  .groupby("station_key")
                  .agg(합쳐진_이름=("name_v", lambda s: " / ".join(sorted(set(s)))),
                       ID=("raw_id", lambda s: " / ".join(sorted(set(s)))),
                       대여건수=("trips", "sum"))
                  .reset_index())
        lines += ["", "이름 변경으로 통합된 대여소:", "", to_markdown(groups)]

    lines += ["", "- 상세: `02_data-eda/check_tables/station_mapping.csv`, `station_key_summary.csv`, "
                  "`station_rename_checks.csv`", ""]

    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main():
    files = sorted(IN_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"통합 데이터가 없습니다: {IN_DIR}\n먼저 00_merge_raw.py를 실행하세요.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("*.parquet"):  # 이전 실행 결과 정리
        p.unlink()

    print("[1/3] 대여소 대응표 생성")
    mapping, pairs, summary, rename_checks = build_station_mapping()
    rename_checks.to_csv(TABLE_DIR / "station_rename_checks.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(TABLE_DIR / "station_mapping.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "station_key_summary.csv", index=False, encoding="utf-8-sig")

    print("[2/3] 내용 중복 확인")
    drop_ids = find_duplicate_trip_ids()

    print(f"[3/3] 파일별 전처리 ({len(files)}개)")
    fallback = dst_fallback_dates(range(PERIOD_START.year, PERIOD_END.year + 1))
    counts = {k: 0 for k in [
        "0. 통합 데이터", "1. 분석 기간 밖", "2. 반납 정보 결측", "3. 종료<시작 (서머타임으로 설명 안 됨)",
        "4. 내용 중복 (사본)", "5. 짧은 왕복 (같은 대여소, 1분 이하)", "6. 긴 대여 (24시간 초과)", "최종 분석 데이터"]}
    ebs_rows = []
    for i, path in enumerate(files, 1):
        process_file(path, mapping, drop_ids, fallback, counts, ebs_rows)
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)} 완료")
        gc.collect()

    ebs_all = pd.concat(ebs_rows) if ebs_rows else None
    if ebs_all is not None:
        ebs_all.to_csv(TABLE_DIR / "end_before_start_rows.csv", index=False, encoding="utf-8-sig")
    write_log(counts, pairs, summary, rename_checks, ebs_all, fallback)

    print(f"\n[완료] 최종 {counts['최종 분석 데이터']:,}행 → {OUT_DIR}")
    print(f"[저장] {LOG_PATH}")


if __name__ == "__main__":
    main()