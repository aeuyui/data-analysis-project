"""
2차 1단계: 통합 데이터 품질 점검 스크립트

목적
- 00_merge_raw.py로 만든 통합 데이터(data/interim/trips/)의 '값'을 점검한다.
- 결측, 논리 오류(종료 < 시작), 대여 시간 이상값, 대여소 ID-이름 대응, 범주 값을 확인해
  다음 단계(전처리)의 기준을 정하는 근거를 만든다.
- 이 스크립트도 데이터를 고치지 않는다. 현황만 보고한다.

폴더 구조 (이 스크립트는 proj 폴더 바로 아래에 둔다)
    proj/
    ├─ 00_merge_raw.py
    ├─ 01_check_data.py     ← 이 파일
    ├─ data/interim/trips/  ← 입력 (통합 결과)
    └─ 02_data-eda/         ← 결과 저장 위치

사용법 (proj 폴더에서 실행)
    python 01_check_data.py

결과
- 02_data-eda/data_check_report.md          : 점검 결과 요약
- 02_data-eda/check_tables/*.csv            : 상세 표 (대여소 ID-이름 대응, 대여 시간 구간별 건수 등)

메모리
- 2,600만 행 규모이므로 점검 항목마다 필요한 컬럼만 읽고, 다 쓰면 메모리에서 지운다.
- 반복되는 문자열(대여소 이름, 자전거 모델 등)은 범주형으로 읽어 메모리를 줄인다.
"""

import gc
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# 경로 기준: 이 스크립트가 있는 폴더(proj)
PROJECT_DIR = Path(__file__).resolve().parent
TRIPS_DIR = PROJECT_DIR / "data" / "interim" / "trips"
REPORT_DIR = PROJECT_DIR / "02_data-eda"
TABLE_DIR = REPORT_DIR / "check_tables"
REPORT_PATH = REPORT_DIR / "data_check_report.md"

# 분석 기간 후보 (완전한 달만 포함). 이 기간 밖의 행 수를 보고한다. 확정은 전처리 단계에서 한다.
PERIOD_START = pd.Timestamp("2023-01-01")
PERIOD_END = pd.Timestamp("2026-01-01")  # 미포함

# 범주형으로 읽을 컬럼 (같은 값이 많이 반복됨)
DICT_COLUMNS = ["start_station_id", "start_station_name", "end_station_id", "end_station_name",
                "bike_id", "bike_model", "duration_source", "source_file"]


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------
def dataset():
    return pq.ParquetDataset(str(TRIPS_DIR), read_dictionary=DICT_COLUMNS)


def read(columns):
    return dataset().read(columns=columns).to_pandas()


def to_markdown(df: pd.DataFrame, index=True) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


def pct(part, whole) -> str:
    return f"{part / whole * 100:.3f}%" if whole else "-"


class Report:
    def __init__(self):
        self.lines = ["# 통합 데이터 품질 점검 결과", "",
                      f"- 실행 일시: {datetime.now():%Y-%m-%d %H:%M}",
                      f"- pandas {pd.__version__}, pyarrow {pa.__version__}",
                      "- 이 보고서는 현황만 기록한다. 데이터는 수정하지 않았다.", ""]

    def h(self, text):
        self.lines += [f"## {text}", ""]
        print(f"\n===== {text} =====")

    def p(self, text=""):
        self.lines.append(text)
        if text:
            print(text)

    def table(self, df, index=True):
        self.lines += [to_markdown(df, index=index), ""]
        print(df.to_string(index=index))

    def save(self):
        REPORT_PATH.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"\n[저장] {REPORT_PATH}")


def save_table(df: pd.DataFrame, name: str, index=False) -> str:
    path = TABLE_DIR / name
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path.relative_to(PROJECT_DIR).as_posix()


# ---------------------------------------------------------------------------
# 1. 규모와 결측
# ---------------------------------------------------------------------------
def check_missing(r: Report) -> int:
    r.h("1. 규모와 결측")
    # 컬럼을 하나씩 읽어 결측 수를 센다 (전체를 한 번에 메모리에 올리지 않음)
    ds = pq.ParquetDataset(str(TRIPS_DIR))
    rows, n = [], None
    for name in ds.schema.names:
        col = ds.read(columns=[name]).column(name)
        n = len(col) if n is None else n
        rows.append({"컬럼": name, "자료형": str(col.type), "결측 수": col.null_count,
                     "결측 비율": pct(col.null_count, len(col))})
        del col
        gc.collect()
    r.p(f"- 전체 행 수: {n:,}")
    r.p()
    r.table(pd.DataFrame(rows), index=False)
    return n


# ---------------------------------------------------------------------------
# 2. 기간과 날짜 논리
# ---------------------------------------------------------------------------
def check_dates(r: Report, n: int):
    r.h("2. 기간과 날짜 논리")
    df = read(["start_time", "end_time", "duration_ms"])

    r.p(f"- 대여 시작 범위: {df['start_time'].min()} ~ {df['start_time'].max()}")
    r.p(f"- 대여 종료 범위: {df['end_time'].min()} ~ {df['end_time'].max()}")
    before = int((df["start_time"] < PERIOD_START).sum())
    after = int((df["start_time"] >= PERIOD_END).sum())
    r.p(f"- 분석 기간 후보({PERIOD_START:%Y-%m-%d} ~ {PERIOD_END - pd.Timedelta(days=1):%Y-%m-%d}) "
        f"이전 시작: {before:,}건 ({pct(before, n)}), 이후 시작: {after:,}건")
    r.p()

    end_before_start = int((df["end_time"] < df["start_time"]).sum())
    same_time = int((df["end_time"] == df["start_time"]).sum())
    r.p(f"- 종료 일시 < 시작 일시: {end_before_start:,}건")
    r.p(f"- 종료 일시 = 시작 일시: {same_time:,}건")

    # 날짜 해상도: 초 단위 값이 있는지 (분 단위로만 기록되었는지 확인)
    has_seconds = int((df["start_time"].dt.second != 0).sum())
    r.p(f"- 시작 일시에 초(second) 값이 0이 아닌 행: {has_seconds:,}건 "
        f"→ {'분 단위로만 기록됨' if has_seconds == 0 else '초 단위 정보 있음'}")
    r.p()

    # duration_ms와 (종료 - 시작)의 일치 여부
    diff_min = ((df["end_time"] - df["start_time"]).dt.total_seconds() * 1000 - df["duration_ms"]) / 60000
    abs_diff = diff_min.abs()
    r.p("대여 시간 컬럼(duration_ms)과 (종료 - 시작)의 차이:")
    bins = pd.Series({
        "1분 이내": int((abs_diff <= 1).sum()),
        "1분 초과 ~ 5분 이내": int(((abs_diff > 1) & (abs_diff <= 5)).sum()),
        "5분 초과 ~ 55분 이내": int(((abs_diff > 5) & (abs_diff <= 55)).sum()),
        "55분 초과 ~ 65분 이내 (서머타임 의심)": int(((abs_diff > 55) & (abs_diff <= 65)).sum()),
        "65분 초과": int((abs_diff > 65).sum()),
        "계산 불가(결측)": int(abs_diff.isna().sum()),
    }).to_frame("건수")
    bins["비율"] = [pct(v, n) for v in bins["건수"]]
    r.table(bins)
    r.p("- 시각이 분 단위로 기록되면 1분 이내 차이는 정상으로 볼 수 있다.")
    r.p("- 60분 근처의 차이가 3월 말·10월 말 특정 날짜에 몰려 있다면 서머타임(BST) 전환 시점의 기록 방식 문제일 수 있으니 아래 날짜 표로 확인한다.")

    big = df.loc[abs_diff > 5, ["start_time", "end_time", "duration_ms"]].copy()
    if not big.empty:
        big["diff_min"] = diff_min[abs_diff > 5].round(1)
        by_day = big["start_time"].dt.date.value_counts().head(15).rename_axis("날짜").to_frame("건수")
        r.p()
        r.p("차이가 5분 넘는 행이 많은 날짜 (상위 15):")
        r.table(by_day)
        path = save_table(big.head(1000), "duration_mismatch_sample.csv")
        r.p(f"- 상세 표본(최대 1,000행): `{path}`")
    r.p()
    del df, diff_min, abs_diff, big
    gc.collect()


# ---------------------------------------------------------------------------
# 3. 대여 시간 분포
# ---------------------------------------------------------------------------
def check_duration(r: Report, n: int):
    r.h("3. 1회 대여 시간 분포 (duration_ms 기준)")
    df = read(["duration_ms", "start_station_id", "end_station_id"])
    minutes = df["duration_ms"] / 60000

    q = minutes.quantile([0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1]).round(2)
    q.index = [f"{x * 100:g}%" for x in q.index]
    r.p("분위수 (분):")
    r.table(q.to_frame("분"))

    edges = [-float("inf"), 0, 1, 2, 5, 60, 180, 24 * 60, float("inf")]
    labels = ["0분 이하", "0~1분", "1~2분", "2~5분", "5분~1시간", "1~3시간", "3~24시간", "24시간 초과"]
    cut = pd.cut(minutes, bins=edges, labels=labels, right=True)
    dist = cut.value_counts().reindex(labels).to_frame("건수")
    dist["비율"] = [pct(v, n) for v in dist["건수"]]
    r.p("구간별 건수 (구간은 오른쪽 끝 포함):")
    r.table(dist)

    # 같은 대여소에서 빌려 바로 반납한 짧은 대여 (잘못 빌린 경우 등일 가능성)
    same = df["start_station_id"].astype("string") == df["end_station_id"].astype("string")
    r.p(f"- 왕복(출발 대여소 = 도착 대여소): {int(same.sum()):,}건 ({pct(int(same.sum()), n)})")
    for th in [1, 2]:
        k = int((same & (minutes <= th)).sum())
        r.p(f"  - 그중 {th}분 이하: {k:,}건 ({pct(k, n)})")
    r.p("- 짧은 왕복은 실제 이동이 아닐 가능성이 있지만, 제외 기준(몇 분 이하를 뺄지)은 이 분포를 근거로 전처리 단계에서 정한다.")
    r.p()
    save_table(dist.reset_index().rename(columns={"index": "구간"}), "duration_bins.csv")
    del df, minutes, cut, same
    gc.collect()


# ---------------------------------------------------------------------------
# 4. 대여소 ID와 이름
# ---------------------------------------------------------------------------
def check_stations(r: Report):
    r.h("4. 대여소 ID와 이름")
    df = read(["start_station_id", "start_station_name", "end_station_id", "end_station_name"])

    # 출발/도착을 합쳐 (ID, 이름) 쌍별 건수
    pairs = pd.concat([
        df.groupby(["start_station_id", "start_station_name"], observed=True).size()
          .rename_axis(["station_id", "station_name"]),
        df.groupby(["end_station_id", "end_station_name"], observed=True).size()
          .rename_axis(["station_id", "station_name"]),
    ]).groupby(level=[0, 1]).sum().rename("건수").reset_index()
    pairs["station_id"] = pairs["station_id"].astype("string")
    pairs["station_name"] = pairs["station_name"].astype("string")

    ids = pairs["station_id"].nunique()
    names = pairs["station_name"].nunique()
    non_numeric = pairs.loc[~pairs["station_id"].str.fullmatch(r"\d+", na=False), "station_id"].unique()
    r.p(f"- 대여소 ID 수: {ids:,}, 대여소 이름 수: {names:,}")
    r.p(f"- 숫자가 아닌 ID: {len(non_numeric)}개 {list(non_numeric[:10])}")

    id_multi = pairs.groupby("station_id")["station_name"].nunique()
    name_multi = pairs.groupby("station_name")["station_id"].nunique()
    r.p(f"- 이름이 2개 이상인 ID: {int((id_multi > 1).sum()):,}개")
    r.p(f"- ID가 2개 이상인 이름: {int((name_multi > 1).sum()):,}개")

    multi = pairs[pairs["station_id"].isin(id_multi[id_multi > 1].index)].sort_values(["station_id", "건수"],
                                                                                    ascending=[True, False])
    if not multi.empty:
        r.p()
        r.p("이름이 여러 개인 ID 예시 (상위 10개 ID):")
        r.table(multi[multi["station_id"].isin(multi["station_id"].unique()[:10])], index=False)

    p1 = save_table(pairs.sort_values(["station_id", "건수"], ascending=[True, False]), "station_id_name_pairs.csv")
    p2 = save_table(pairs[pairs["station_name"].isin(name_multi[name_multi > 1].index)]
                    .sort_values(["station_name", "건수"], ascending=[True, False]), "station_name_multi_id.csv")
    r.p(f"- 전체 (ID, 이름) 쌍: `{p1}`")
    r.p(f"- ID가 여러 개인 이름: `{p2}`")
    r.p("- 분석 단위를 ID로 할지 이름으로 할지는 이 표를 보고 전처리 단계에서 정한다.")
    r.p()
    del df, pairs
    gc.collect()


# ---------------------------------------------------------------------------
# 5. 범주 값과 연도별 구성
# ---------------------------------------------------------------------------
def check_categories(r: Report, n: int):
    r.h("5. 자전거 모델과 연도별 구성")
    df = read(["start_time", "bike_model", "bike_id"])
    year = df["start_time"].dt.year

    model = df["bike_model"].astype("string").fillna("<결측>")
    r.p("연도별 자전거 모델 건수:")
    r.table(pd.crosstab(model, year, margins=True, margins_name="합계"))
    r.p(f"- 자전거 ID 수: {df['bike_id'].nunique():,}")
    r.p()
    del df, year, model
    gc.collect()


# ---------------------------------------------------------------------------
# 6. 같은 대여가 다른 ID로 두 번 기록되었는지
# ---------------------------------------------------------------------------
def check_logical_duplicates(r: Report, n: int):
    r.h("6. 내용 중복 (trip_id는 다르지만 같은 대여로 보이는 행)")
    cols = ["bike_id", "start_time", "end_time", "start_station_id", "end_station_id"]
    df = read(cols)
    dup = df.duplicated(subset=cols, keep=False)
    k = int(dup.sum())
    r.p(f"- 기준: {', '.join(cols)} 가 모두 같은 행")
    r.p(f"- 해당 행 수: {k:,} ({pct(k, n)})")
    if k:
        path = save_table(df[dup].sort_values(cols).head(1000), "logical_duplicates_sample.csv")
        r.p(f"- 표본(최대 1,000행): `{path}`")
    r.p("- 같은 자전거가 같은 시각에 같은 대여소에서 두 번 출발할 수는 없으므로, 해당 행이 있으면 중복 기록 후보로 본다.")
    r.p()
    del df, dup
    gc.collect()


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main():
    if not TRIPS_DIR.exists() or not any(TRIPS_DIR.glob("*.parquet")):
        raise FileNotFoundError(f"통합 데이터가 없습니다: {TRIPS_DIR}\n먼저 00_merge_raw.py를 실행하세요.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    r = Report()
    n = check_missing(r)
    check_dates(r, n)
    check_duration(r, n)
    check_stations(r)
    check_categories(r, n)
    check_logical_duplicates(r, n)
    r.h("7. 다음 단계")
    r.p("- 위 결과를 근거로 전처리 기준(분석 기간, 대여 시간 이상값, 짧은 왕복, 대여소 식별 단위, 중복)을 정한다.")
    r.save()


if __name__ == "__main__":
    main()
