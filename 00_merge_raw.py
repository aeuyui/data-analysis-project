"""
2차 0단계: TfL 원데이터(여러 파일) 통합 스크립트

목적
- data/raw/ 아래의 TfL 공공자전거 이용 기록 파일(CSV, XLSX)을 하나의 형식으로 통합한다.
- 이 단계에서는 '형식'만 통일하고 '값'은 고치지 않는다.
  · 행을 삭제하지 않는다 (파일별 읽은 행 수 = 저장한 행 수를 자동 확인).
  · 결측, 중복, 이상값은 '보고'만 하고, 처리 기준은 다음 단계(전처리)에서 정한다.

통일하는 것
1) 컬럼명: 시기별로 다른 컬럼명(예: 'Rental Id' / 'Number')을 표준 이름으로 바꾼다.
2) 날짜: 파일마다 날짜 형식을 판별해 datetime으로 변환한다.
   (영국 데이터라 '01/08/2023' 같은 일/월 형식이 많으므로 형식을 자동 추측하지 않고 후보 형식으로 판별한다.)
3) 대여 시간: 초 단위('Duration')와 밀리초 단위('Total duration (ms)')를 duration_ms 하나로 맞춘다.
4) 출처: 각 행에 source_file(원본 파일명)을 붙여 어느 파일에서 왔는지 추적할 수 있게 한다.

폴더 구조 (이 스크립트는 proj 폴더 바로 아래에 둔다)
    proj/
    ├─ 00_merge_raw.py      ← 이 파일
    ├─ data/
    │  ├─ raw/              ← TfL 원데이터 (하위 폴더도 읽음)
    │  └─ interim/          ← 통합 결과 저장 위치
    ├─ 02_data-eda/         ← 처리 기록·보고서 저장 위치 (없으면 자동 생성)
    └─ venv/

사용법 (proj 폴더에서 실행)
    pip install pandas pyarrow openpyxl tabulate
    python 00_merge_raw.py

    ※ 경로는 이 파일의 위치를 기준으로 계산하므로, 다른 폴더에서 실행해도 같은 위치를 사용한다.

결과
- data/interim/trips/<원본파일명>.parquet : 원본 파일 1개당 parquet 1개 (표준 컬럼)
- 02_data-eda/merge_manifest.csv          : 파일별 처리 기록 (행 수, 형식, 날짜 형식, 변환 실패 수 등)
- 02_data-eda/merge_report.md             : 통합 결과 요약 (월별 행 수, 중복, 파일 간 기간 겹침 등)

parquet을 쓰는 이유
- 3년치 CSV는 수천만 행이라 용량과 메모리 부담이 크다.
- parquet은 압축률이 높고, 분석할 때 필요한 컬럼만 골라 읽을 수 있다.
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# 경로 기준: 이 스크립트가 있는 폴더(proj)
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUT_DIR = PROJECT_DIR / "data" / "interim" / "trips"
REPORT_DIR = PROJECT_DIR / "02_data-eda"
MANIFEST_PATH = REPORT_DIR / "merge_manifest.csv"
REPORT_PATH = REPORT_DIR / "merge_report.md"

# ---------------------------------------------------------------------------
# 표준 컬럼과 원본 컬럼명 대응표
#   키: 표준 컬럼명 / 값: 원본에서 나타날 수 있는 컬럼명(소문자, 공백 정리 기준)
#   실제 파일에서 '대응되지 않은 컬럼'이 나오면 보고서에 표시되므로, 그때 여기에 추가한다.
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "trip_id": ["number", "rental id"],
    "start_time": ["start date"],
    "start_station_id": ["start station number", "startstation id", "start station id"],
    "start_station_name": ["start station", "startstation name", "start station name"],
    "end_time": ["end date"],
    "end_station_id": ["end station number", "endstation id", "end station id"],
    "end_station_name": ["end station", "endstation name", "end station name"],
    "bike_id": ["bike number", "bike id"],
    "bike_model": ["bike model"],
    "duration_text": ["total duration"],
    "duration_ms_raw": ["total duration (ms)"],
    "duration_s_raw": ["duration", "duration_seconds"],
}

STANDARD_COLUMNS = [
    "trip_id",
    "start_time",
    "start_station_id",
    "start_station_name",
    "end_time",
    "end_station_id",
    "end_station_name",
    "bike_id",
    "bike_model",
    "duration_ms",
    "duration_source",
    "duration_text",
    "source_file",
]

# 모든 parquet 파일이 같은 스키마를 갖도록 고정한다.
# (station_id를 문자열로 두는 이유: 과거 데이터에 숫자가 아닌 대여소 ID가 있었다는 보고가 있어,
#  값을 바꾸지 않고 보존하기 위함. 숫자 변환은 전처리 단계에서 결정한다.)
SCHEMA = pa.schema(
    [
        ("trip_id", pa.int64()),
        ("start_time", pa.timestamp("ns")),
        ("start_station_id", pa.string()),
        ("start_station_name", pa.string()),
        ("end_time", pa.timestamp("ns")),
        ("end_station_id", pa.string()),
        ("end_station_name", pa.string()),
        ("bike_id", pa.string()),
        ("bike_model", pa.string()),
        ("duration_ms", pa.float64()),
        ("duration_source", pa.string()),
        ("duration_text", pa.string()),
        ("source_file", pa.string()),
    ]
)

# 날짜 후보 형식. 일/월(영국식)을 월/일보다 먼저 둔다.
DATE_FORMATS = [
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
]
DATE_SAMPLE_SIZE = 5000


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------
def normalize(name) -> str:
    """컬럼명 비교용: 대소문자, 앞뒤 공백, 연속 공백, BOM 차이를 무시한다."""
    return " ".join(str(name).replace("﻿", "").strip().lower().split())


def build_rename_map(columns):
    """원본 컬럼 → 표준 컬럼 대응을 만든다. 대응되지 않은 컬럼 목록도 함께 반환한다."""
    alias_to_std = {alias: std for std, aliases in COLUMN_ALIASES.items() for alias in aliases}
    rename, unmapped = {}, []
    for col in columns:
        std = alias_to_std.get(normalize(col))
        if std is None:
            unmapped.append(str(col))
        elif std in rename.values():
            unmapped.append(f"{col} (표준 컬럼 '{std}' 중복 대응)")
        else:
            rename[col] = std
    return rename, unmapped


def read_raw(path: Path) -> pd.DataFrame:
    """값을 바꾸지 않도록 모든 컬럼을 문자열로 읽는다."""
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="latin-1")
    return pd.read_excel(path, dtype=str)


def detect_date_format(values: pd.Series):
    """
    표본 값에 대해 후보 형식별 변환 성공률을 구해 가장 높은 형식을 고른다.
    반환: (선택 형식, 성공률, 모호 여부)
      - 모호: 일/월 형식과 월/일 형식이 둘 다 표본 전체를 변환할 수 있는 경우
              (표본의 '일'이 모두 12 이하일 때 발생). 이때는 영국식(일/월)을 선택하고 기록한다.
    """
    sample = values.dropna().astype(str).str.strip()
    sample = sample[sample != ""]
    if sample.empty:
        return None, 0.0, False
    sample = sample.sample(min(len(sample), DATE_SAMPLE_SIZE), random_state=0)

    rates = {}
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        rates[fmt] = parsed.notna().mean()

    best = max(DATE_FORMATS, key=lambda f: rates[f])  # 동률이면 앞(일/월)의 형식이 선택됨
    ambiguous = False
    if best.startswith("%d/%m"):
        mirror = best.replace("%d/%m", "%m/%d")
        ambiguous = rates.get(mirror, 0) >= rates[best]
    return best, rates[best], ambiguous


def parse_dates(values: pd.Series, fmt):
    if fmt is None:
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    return pd.to_datetime(values.str.strip(), format=fmt, errors="coerce").astype("datetime64[ns]")


def count_failed(original: pd.Series, converted: pd.Series) -> int:
    """원본에 값이 있었는데 변환 후 결측이 된 개수 (= 변환 실패)."""
    return int((original.notna() & converted.isna()).sum())


# ---------------------------------------------------------------------------
# 파일 1개 처리
# ---------------------------------------------------------------------------
def process_file(path: Path) -> dict:
    raw = read_raw(path)
    rows_read = len(raw)
    rename, unmapped = build_rename_map(raw.columns)
    df = raw.rename(columns=rename)
    record = {
        "file": path.relative_to(RAW_DIR).as_posix(),
        "rows_read": rows_read,
        "original_columns": " | ".join(map(str, raw.columns)),
        "unmapped_columns": " | ".join(unmapped),
    }

    out = pd.DataFrame(index=df.index)

    # trip_id: 정수 변환 (실패 수 기록)
    trip_raw = df.get("trip_id", pd.Series(pd.NA, index=df.index, dtype="object"))
    out["trip_id"] = pd.to_numeric(trip_raw, errors="coerce").astype("Int64")
    record["trip_id_parse_fail"] = count_failed(trip_raw, out["trip_id"])

    # 날짜: 파일별 형식 판별 후 변환
    for col in ["start_time", "end_time"]:
        raw_col = df.get(col, pd.Series(pd.NA, index=df.index, dtype="object"))
        fmt, rate, ambiguous = detect_date_format(raw_col)
        out[col] = parse_dates(raw_col.astype("string"), fmt)
        prefix = col.replace("_time", "")
        record[f"{prefix}_date_format"] = fmt or ""
        record[f"{prefix}_date_sample_rate"] = round(rate, 4)
        record[f"{prefix}_date_ambiguous"] = ambiguous
        record[f"{prefix}_date_parse_fail"] = count_failed(raw_col, out[col])

    # 문자열 컬럼: 값 그대로 (앞뒤 공백만 제거)
    for col in ["start_station_id", "start_station_name", "end_station_id", "end_station_name",
                "bike_id", "bike_model", "duration_text"]:
        if col in df.columns:
            out[col] = df[col].astype("string").str.strip()
        else:
            out[col] = pd.Series(pd.NA, index=df.index, dtype="string")

    # 대여 시간: ms 컬럼 우선, 없으면 초 단위 × 1000
    if "duration_ms_raw" in df.columns:
        out["duration_ms"] = pd.to_numeric(df["duration_ms_raw"], errors="coerce")
        out["duration_source"] = "ms"
        record["duration_parse_fail"] = count_failed(df["duration_ms_raw"], out["duration_ms"])
    elif "duration_s_raw" in df.columns:
        out["duration_ms"] = pd.to_numeric(df["duration_s_raw"], errors="coerce") * 1000
        out["duration_source"] = "seconds"
        record["duration_parse_fail"] = count_failed(df["duration_s_raw"], out["duration_ms"])
    else:
        out["duration_ms"] = float("nan")
        out["duration_source"] = "none"
        record["duration_parse_fail"] = 0
    out["duration_source"] = out["duration_source"].astype("string")

    out["source_file"] = record["file"]
    out["source_file"] = out["source_file"].astype("string")
    out = out[STANDARD_COLUMNS]

    # 이 파일에 없어서 전부 결측인 표준 컬럼 (예: 과거 형식에는 bike_model 없음)
    absent = [c for c in STANDARD_COLUMNS if c not in ("duration_ms", "duration_source", "source_file")
              and out[c].isna().all()]
    record["absent_standard_columns"] = " | ".join(absent)
    record["schema_type"] = (
        "new (ms)" if "duration_ms_raw" in df.columns
        else "old (seconds)" if "duration_s_raw" in df.columns
        else "unknown"
    )

    # 통합 단계에서는 행을 삭제하지 않는다.
    assert len(out) == rows_read, f"{path.name}: 행 수가 달라졌습니다 ({rows_read} → {len(out)})"
    record["rows_written"] = len(out)
    record["start_min"] = str(out["start_time"].min())
    record["start_max"] = str(out["start_time"].max())

    out_path = OUT_DIR / (safe_name(record["file"]) + ".parquet")
    table = pa.Table.from_pandas(out, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, out_path, compression="zstd")
    record["output"] = out_path.relative_to(PROJECT_DIR).as_posix()
    return record


def safe_name(relative: str) -> str:
    stem = re.sub(r"\.(csv|xlsx|xls)$", "", relative, flags=re.IGNORECASE)
    return re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", stem)


# ---------------------------------------------------------------------------
# 통합 결과 점검 (필요한 컬럼만 읽어 메모리 절약)
# ---------------------------------------------------------------------------
def to_markdown(df: pd.DataFrame, index=True) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


def build_report(manifest: pd.DataFrame):
    lines = ["# 원데이터 통합 결과", "", f"- 실행 일시: {datetime.now():%Y-%m-%d %H:%M}",
             f"- pandas {pd.__version__}, pyarrow {pa.__version__}", ""]

    # 1. 파일 요약
    lines += ["## 1. 파일 요약", ""]
    lines.append(f"- 처리한 파일 수: {len(manifest)}")
    lines.append(f"- 읽은 행 수 합계: {manifest['rows_read'].sum():,}")
    lines.append(f"- 저장한 행 수 합계: {manifest['rows_written'].sum():,} (통합 단계에서는 삭제 없음)")
    lines.append("")
    lines.append("형식별 파일 수:")
    lines.append(to_markdown(manifest["schema_type"].value_counts().to_frame("파일 수")))
    lines.append("")

    # 2. 확인이 필요한 파일
    lines += ["## 2. 확인이 필요한 파일", ""]
    checks = {
        "대응되지 않은 컬럼이 있는 파일": manifest["unmapped_columns"].fillna("") != "",
        "형식을 알 수 없는 파일(대여 시간 컬럼 없음)": manifest["schema_type"] == "unknown",
        "날짜 형식이 모호했던 파일(일/월로 처리)": manifest["start_date_ambiguous"] | manifest["end_date_ambiguous"],
        "날짜 변환 실패가 있는 파일": (manifest["start_date_parse_fail"] + manifest["end_date_parse_fail"]) > 0,
        "trip_id 변환 실패가 있는 파일": manifest["trip_id_parse_fail"] > 0,
        "대여 시간 변환 실패가 있는 파일": manifest["duration_parse_fail"] > 0,
    }
    for label, mask in checks.items():
        files = manifest.loc[mask, "file"].tolist()
        lines.append(f"- {label}: {len(files)}개")
        for f in files[:20]:
            lines.append(f"  - `{f}`")
        if len(files) > 20:
            lines.append(f"  - … 외 {len(files) - 20}개 (merge_manifest.csv 참고)")
    unmapped = manifest.loc[manifest["unmapped_columns"].fillna("") != "", "unmapped_columns"]
    if not unmapped.empty:
        lines.append("")
        lines.append("대응되지 않은 컬럼 목록 (00_merge_raw.py의 COLUMN_ALIASES에 추가할지 판단 필요):")
        for v in sorted(set(" | ".join(unmapped).split(" | "))):
            lines.append(f"- `{v}`")
    lines.append("")

    # 3. 월별 행 수 (start_time 기준)
    # source_file은 같은 값이 수백만 번 반복되므로 범주형(dictionary)으로 읽어 메모리를 줄인다.
    dataset = pq.ParquetDataset(str(OUT_DIR), read_dictionary=["source_file"])
    times = dataset.read(columns=["start_time", "source_file"]).to_pandas()
    month = times["start_time"].dt.to_period("M").astype("string").fillna("변환 실패")
    monthly = month.value_counts().sort_index().to_frame("행 수")
    lines += ["## 3. 월별 행 수 (대여 시작 기준)", "",
              "빠진 달이나 행 수가 유난히 적은 달은 파일 누락 가능성이 있으니 확인한다.", ""]
    lines.append(to_markdown(monthly))
    lines.append("")
    aug_2023 = int((month == "2023-08").sum())
    lines.append(f"- (검증용) 2023년 8월 행 수: {aug_2023:,} — 1차 기획서의 Kaggle 데이터(776,527행)와 비교한다. "
                 "단, 파일 경계(주·월 단위)에 걸친 대여의 포함 방식 때문에 정확히 같지 않을 수 있다.")
    lines.append("")

    # 4. 파일 간 기간 겹침
    lines += ["## 4. 파일 간 기간 겹침", ""]
    span = manifest[["file", "start_min", "start_max"]].copy()
    span["start_min"] = pd.to_datetime(span["start_min"], errors="coerce")
    span["start_max"] = pd.to_datetime(span["start_max"], errors="coerce")
    span = span.dropna().sort_values("start_min").reset_index(drop=True)
    overlaps = [
        (span.loc[i, "file"], span.loc[i + 1, "file"])
        for i in range(len(span) - 1)
        if span.loc[i + 1, "start_min"] < span.loc[i, "start_max"]
    ]
    lines.append(f"- 이웃한 파일과 기간이 겹치는 경우: {len(overlaps)}건 (중복 대여가 들어 있을 수 있음)")
    for a, b in overlaps[:20]:
        lines.append(f"  - `{a}` ↔ `{b}`")
    lines.append("")

    # 5. 중복 (trip_id 기준)
    ids = dataset.read(columns=["trip_id", "source_file"]).to_pandas()
    dup_mask = ids["trip_id"].notna() & ids["trip_id"].duplicated(keep=False)
    dup = ids[dup_mask]
    cross_file = dup.groupby("trip_id")["source_file"].nunique()
    lines += ["## 5. 중복 (trip_id 기준)", ""]
    lines.append(f"- trip_id 결측: {int(ids['trip_id'].isna().sum()):,}")
    lines.append(f"- 2번 이상 나타나는 trip_id 수: {int(dup['trip_id'].nunique()):,}")
    lines.append(f"- 그중 서로 다른 파일에 걸친 trip_id 수: {int((cross_file > 1).sum()):,}")
    lines.append(f"- 중복에 해당하는 전체 행 수: {len(dup):,}")
    lines.append("- 처리하지 않았음. 다음 단계에서 '완전히 같은 행인지, 내용이 다른지' 확인 후 기준을 정한다.")
    lines.append("")
    lines.append("## 6. 다음 단계")
    lines.append("")
    lines.append("- 대응되지 않은 컬럼, 모호한 날짜 형식, 누락 의심 월을 먼저 확인한다.")
    lines.append("- 통합 데이터(data/interim/trips/)를 대상으로 결측·중복·이상값을 점검하고 전처리 기준을 정한다.")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TfL 원데이터 통합")
    parser.add_argument("--raw-dir", type=Path, default=None, help="원본 폴더 (기본: data/raw)")
    args = parser.parse_args()

    global RAW_DIR
    if args.raw_dir is not None:
        RAW_DIR = args.raw_dir.resolve()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"원본 폴더가 없습니다: {RAW_DIR}")

    # 엑셀이 열려 있을 때 생기는 임시 파일(~$...)과 숨김 파일은 제외한다.
    files = sorted(
        p for p in RAW_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (".csv", ".xlsx", ".xls")
        and not p.name.startswith(("~$", "."))
    )
    if not files:
        raise FileNotFoundError(f"{RAW_DIR} 아래에 CSV/XLSX 파일이 없습니다.")

    print(f"[경로] 프로젝트: {PROJECT_DIR}")
    print(f"[경로] 원본: {RAW_DIR}")
    print(f"[경로] 보고서: {REPORT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    old = list(OUT_DIR.glob("*.parquet"))
    for p in old:  # 이전 실행 결과가 섞이지 않도록 지운 뒤 새로 만든다 (원본은 건드리지 않음)
        p.unlink()

    print(f"[시작] 파일 {len(files)}개 → {OUT_DIR}")
    records = []
    for i, path in enumerate(files, 1):
        try:
            rec = process_file(path)
            print(f"  [{i}/{len(files)}] {rec['file']}: {rec['rows_read']:,}행, 형식 {rec['schema_type']}")
        except Exception as e:  # 한 파일 오류로 전체가 멈추지 않도록 기록 후 계속
            rec = {"file": path.relative_to(RAW_DIR).as_posix(), "error": repr(e)}
            print(f"  [{i}/{len(files)}] {rec['file']}: 오류 — {e}")
        records.append(rec)

    manifest = pd.DataFrame(records)
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    print(f"[저장] {MANIFEST_PATH}")

    if "error" in manifest.columns and manifest["error"].notna().any():
        print("[중단] 오류가 난 파일이 있어 보고서를 만들지 않았습니다. merge_manifest.csv의 error 열을 확인하세요.")
        return

    build_report(manifest)
    print(f"[저장] {REPORT_PATH}")


if __name__ == "__main__":
    main()