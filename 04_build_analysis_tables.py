"""
3차 1단계: 분석용 테이블 만들기

목적
- 질문 1·2에 필요한 '대여소 × 연도 × 근무일 여부 × 시간대'의 출발·도착 건수를 만든다.
  (EDA의 집계에는 연도 구분이 없어, 연도별 안정성 확인을 위해 새로 만든다.)
- 전처리 기준 민감도 확인용으로 같은 표를 기준을 바꾼 두 가지 변형으로도 만든다.
    base   : 현재 전처리 결과 그대로
    short2 : 같은 대여소 왕복 '2분 이하'까지 제외 (현재 기준 1분 이하)
    long3h : '3시간 초과' 대여까지 제외 (현재 기준 24시간 초과)
  ※ 두 변형은 현재 데이터에서 '더 빼는' 방향만 가능하다. 덜 빼는 방향은 중간 데이터로 다시 전처리해야 한다.
- 질문 3(대여 시간 모델)에 쓸 표본(약 5%)을 뽑는다. trip_id로 뽑으므로 다시 실행해도 같은 표본이 나온다.
- 운영 장애일(2025-08-05)은 모든 분석 테이블에서 제외한다.

폴더 구조 (proj 폴더 바로 아래에 둔다. common.py 필요)
    입력: data/processed/trips/
    출력: data/agg/station_hourly_year.parquet, data/agg/day_counts.parquet,
          data/processed/q3_sample.parquet, 03_analysis-modeling/tables_log.md

사용법 (proj 폴더에서 실행)
    python 04_build_analysis_tables.py
"""

import gc
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import (AGG_DIR, HOLIDAY_DATES, OUT_DIR, OUTAGE_DATES, OUTAGE_DAYS, PROCESSED_DIR, TRIPS_DIR,
                    day_counts, fmt_int, to_md)

SAMPLE_MOD = 20  # trip_id % 20 == 0 → 약 5% 표본
VARIANTS = {
    "base": "현재 전처리 결과",
    "short2": "같은 대여소 왕복 2분 이하까지 추가 제외",
    "long3h": "3시간 초과 대여까지 추가 제외",
}


def main():
    files = sorted(TRIPS_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"분석용 데이터가 없습니다: {TRIPS_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)

    cols = ["trip_id", "start_time", "end_time", "duration_min", "start_station_key", "end_station_key",
            "is_round_trip", "bike_model", "year", "month", "season", "start_date", "start_hour", "day_of_week"]
    store, codes = {}, {}
    rows = {"read": 0, "outage": 0}
    variant_rows = {v: 0 for v in VARIANTS}
    samples = []

    for i, path in enumerate(files, 1):
        df = pq.read_table(path, columns=cols).to_pandas()
        rows["read"] += len(df)
        outage = df["start_date"].isin(OUTAGE_DATES)
        rows["outage"] += int(outage.sum())
        df = df[~outage].copy()

        for k in pd.unique(pd.concat([df["start_station_key"], df["end_station_key"]], ignore_index=True)):
            codes.setdefault(k, len(codes))
        df["s"] = df["start_station_key"].map(codes).astype("int32")
        df["e"] = df["end_station_key"].map(codes).astype("int32")
        df["rt"] = df["is_round_trip"].astype(bool)
        df["is_holiday"] = df["start_date"].isin(HOLIDAY_DATES)
        df["workday"] = (df["day_of_week"] < 5) & ~df["is_holiday"]
        df["end_hour"] = df["end_time"].dt.hour.astype("int8")
        df["year"] = df["year"].astype("int16")

        masks = {
            "base": np.ones(len(df), dtype=bool),
            "short2": ~(df["rt"] & (df["duration_min"] <= 2)).to_numpy(),
            "long3h": ~(df["duration_min"] > 180).to_numpy(),
        }
        for v, m in masks.items():
            sub = df[m]
            variant_rows[v] += len(sub)
            for name, key, hour in [("starts", "s", "start_hour"), ("ends", "e", "end_hour")]:
                g = sub.groupby([key, "year", "workday", hour]).size()
                g.index.names = ["code", "year", "workday", "hour"]
                k = (v, name)
                store[k] = g if k not in store else store[k].add(g, fill_value=0)

        smp = df[df["trip_id"] % SAMPLE_MOD == 0]
        samples.append(smp[["trip_id", "start_time", "duration_min", "start_station_key", "end_station_key",
                            "is_round_trip", "bike_model", "year", "month", "season", "start_date",
                            "start_hour", "day_of_week", "is_holiday", "workday"]])
        del df
        gc.collect()
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    # 대여소 × 연도 × 근무일 × 시간대 표
    names = pd.Series({v: k for k, v in codes.items()})
    parts = []
    for v in VARIANTS:
        t = pd.concat([store[(v, "starts")].rename("starts"), store[(v, "ends")].rename("ends")], axis=1)
        t = t.fillna(0).astype("int64").reset_index()
        t.insert(0, "variant", v)
        parts.append(t)
    sh = pd.concat(parts, ignore_index=True)
    sh.insert(1, "station_key", sh["code"].map(names))
    sh = sh.drop(columns="code")
    sh["hour"] = sh["hour"].astype("int8")
    pq.write_table(pa.Table.from_pandas(sh, preserve_index=False), AGG_DIR / "station_hourly_year.parquet",
                   compression="zstd")
    dc = day_counts()
    dc.to_parquet(AGG_DIR / "day_counts.parquet", index=False)

    sample = pd.concat(samples, ignore_index=True)
    sample["day_type"] = np.where(sample["is_holiday"], "공휴일",
                                  np.where(sample["day_of_week"] >= 5, "주말", "평일"))
    pq.write_table(pa.Table.from_pandas(sample, preserve_index=False), PROCESSED_DIR / "q3_sample.parquet",
                   compression="zstd")

    # 기록과 검산
    base = sh[sh["variant"] == "base"]
    checks = {
        "읽은 행": rows["read"],
        "운영 장애일 제외": rows["outage"],
        "분석 행 (base)": variant_rows["base"],
        "표의 출발 합계 (base)": int(base["starts"].sum()),
        "표의 도착 합계 (base)": int(base["ends"].sum()),
    }
    ok = (checks["분석 행 (base)"] == checks["표의 출발 합계 (base)"] == checks["표의 도착 합계 (base)"]
          == rows["read"] - rows["outage"])
    vtab = pd.DataFrame({
        "변형": list(VARIANTS), "설명": list(VARIANTS.values()),
        "행 수": [fmt_int(variant_rows[v]) for v in VARIANTS],
        "base 대비": [f"{(variant_rows[v] - variant_rows['base']) / variant_rows['base'] * 100:+.3f}%" for v in VARIANTS],
    })
    lines = ["# 3차 분석용 테이블 생성 기록", "",
             f"- 실행 일시: {datetime.now():%Y-%m-%d %H:%M}",
             f"- 운영 장애일 제외: {', '.join(f'{d} ({r})' for d, r in OUTAGE_DAYS.items())}", "",
             "## 검산", "",
             to_md(pd.DataFrame({"항목": checks.keys(), "건수": [fmt_int(v) for v in checks.values()]})), "",
             f"- 결과: **{'일치' if ok else '불일치 — 확인 필요'}** (분석 행 = 읽은 행 − 장애일 = 표의 출발 합계 = 도착 합계)", "",
             "## 전처리 기준 민감도용 변형", "", to_md(vtab), "",
             "## 질문 3 표본", "",
             f"- 추출 규칙: trip_id가 {SAMPLE_MOD}의 배수인 대여 (재실행해도 같은 표본)",
             f"- 표본 크기: {len(sample):,}행 (분석 행의 {len(sample) / variant_rows['base'] * 100:.2f}%)",
             "- 연도별 표본: " + ", ".join(f"{y}년 {n:,}행" for y, n in sample["year"].value_counts().sort_index().items()),
             "", "## 날짜 수 (하루 평균의 분모, 장애일 제외)", "",
             to_md(dc.assign(workday=dc["workday"].map({True: "근무일", False: "비근무일"}))), ""]
    (OUT_DIR / "tables_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[완료] 검산 {'일치' if ok else '불일치'} → {OUT_DIR / 'tables_log.md'}")


if __name__ == "__main__":
    main()
