"""
2차 3단계: 탐색적 데이터 분석(EDA)

목적
- 전처리된 분석용 데이터(data/processed/trips/)를 집계해 이용 패턴을 탐색한다.
- 2,600만 행을 한 번에 메모리에 올리지 않도록, 파일별로 집계해 누적한다.
- 집계 테이블은 data/agg/에 저장해 3차 분석(대여소 유형, 흐름, 대여 시간)과 4차 앱에서 다시 쓴다.
- 보고서에는 '관찰(수치)'만 적는다. 해석은 그래프와 표를 보고 사람이 따로 작성한다.

날짜 구분
- 공휴일: 잉글랜드·웨일스 bank holiday (출처: GOV.UK, https://www.gov.uk/bank-holidays, 2026-11-16 확인)
- 근무일(workday): 월~금 중 공휴일이 아닌 날 / 비근무일: 토·일·공휴일

폴더 구조 (이 스크립트는 proj 폴더 바로 아래에 둔다)
    proj/
    ├─ 03_eda.py                 ← 이 파일
    ├─ data/processed/trips/     ← 입력 (02_preprocess.py 결과)
    ├─ data/agg/                 ← 출력: 집계 테이블 (용량이 작아 GitHub에 올려도 됨)
    ├─ data/external/            ← 출력: 공휴일 목록
    └─ 02_data-eda/
       ├─ eda_report.md          ← 출력: 관찰 수치 보고서
       └─ figures/               ← 출력: 그래프 PNG

사용법 (proj 폴더에서 실행)
    pip install matplotlib
    python 03_eda.py
"""

import gc
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
IN_DIR = PROJECT_DIR / "data" / "processed" / "trips"
AGG_DIR = PROJECT_DIR / "data" / "agg"
EXT_DIR = PROJECT_DIR / "data" / "external"
REPORT_DIR = PROJECT_DIR / "02_data-eda"
FIG_DIR = REPORT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "eda_report.md"

PERIOD_START, PERIOD_END = "2023-01-01", "2025-12-31"

# 잉글랜드·웨일스 bank holiday (GOV.UK). 2023-05-08은 국왕 대관식 임시 공휴일.
BANK_HOLIDAYS = {
    "2023-01-02": "New Year's Day (substitute)", "2023-04-07": "Good Friday", "2023-04-10": "Easter Monday",
    "2023-05-01": "Early May", "2023-05-08": "Coronation of King Charles III", "2023-05-29": "Spring",
    "2023-08-28": "Summer", "2023-12-25": "Christmas Day", "2023-12-26": "Boxing Day",
    "2024-01-01": "New Year's Day", "2024-03-29": "Good Friday", "2024-04-01": "Easter Monday",
    "2024-05-06": "Early May", "2024-05-27": "Spring", "2024-08-26": "Summer",
    "2024-12-25": "Christmas Day", "2024-12-26": "Boxing Day",
    "2025-01-01": "New Year's Day", "2025-04-18": "Good Friday", "2025-04-21": "Easter Monday",
    "2025-05-05": "Early May", "2025-05-26": "Spring", "2025-08-25": "Summer",
    "2025-12-25": "Christmas Day", "2025-12-26": "Boxing Day",
}
HOLIDAY_DATES = pd.to_datetime(list(BANK_HOLIDAYS))

WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]
SEASON_OF = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
             6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}
AM_PEAK, PM_PEAK = range(7, 10), range(16, 19)   # 7~9시대, 16~18시대
DUR_BIN = 0.1                                     # 대여 시간 히스토그램 간격(분)

# ---------------------------------------------------------------------------
# 그래프 스타일 (dataviz 기본 팔레트, 라이트 모드)
# ---------------------------------------------------------------------------
C = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",   # 범주 1~3 (검증 통과)
    "s1_light": "#9ec5f4", "neg": "#e34948",             # 보조 / 발산(음수)
}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def setup_font():
    """설치된 한글 글꼴을 찾아 쓴다 (Windows: 맑은 고딕, macOS: AppleGothic 등)."""
    candidates = ["Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic",
                  "Noto Sans CJK KR", "Noto Sans KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[경고] 한글 글꼴을 찾지 못했습니다. 그래프의 한글이 깨질 수 있습니다.")
    plt.rcParams["axes.unicode_minus"] = False


def new_ax(width=9, height=4.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(C["surface"])
    ax.set_facecolor(C["surface"])
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(C["axis"])
    ax.tick_params(colors=C["ink2"], labelsize=9, length=0)
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def finish(fig, ax, title, subtitle, filename, xlabel=None, ylabel=None):
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=12, color=C["ink"], fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9, color=C["ink2"], pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=C["ink2"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["ink2"], fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, facecolor=C["surface"])
    plt.close(fig)
    return f"figures/{filename}"


def end_labels(ax, items):
    """선 끝 직접 라벨 (색만으로 계열을 구분하지 않도록). 라벨끼리 겹치면 위아래로 벌린다.
    items: [(x, y, text, color), ...]"""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.07
    xpad = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
    placed = []
    for x, y, text, color in sorted(items, key=lambda it: it[1]):
        ty = y if not placed or y - placed[-1] >= gap else placed[-1] + gap
        placed.append(ty)
        ax.plot([x], [y], marker="o", markersize=4, color=color)
        ax.text(x + xpad, ty, text, va="center", fontsize=9, color=C["ink2"], clip_on=False)


def fmt_int(x):
    return f"{x:,.0f}"


def to_md(df: pd.DataFrame, index=False) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


# ---------------------------------------------------------------------------
# 1) 파일별 집계 → 누적
# ---------------------------------------------------------------------------
class Acc:
    def __init__(self):
        self.store = {}

    def add(self, name, s: pd.Series):
        s = s.astype("int64")
        self.store[name] = s if name not in self.store else self.store[name].add(s, fill_value=0).astype("int64")


def aggregate():
    files = sorted(IN_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"분석용 데이터가 없습니다: {IN_DIR}\n먼저 02_preprocess.py를 실행하세요.")
    cols = ["end_time", "duration_min", "start_station_key", "end_station_key", "is_round_trip",
            "bike_model", "year", "start_date", "start_hour", "day_of_week"]
    acc, codes, total_rows = Acc(), {}, 0

    for i, path in enumerate(files, 1):
        df = pq.read_table(path, columns=cols).to_pandas()
        total_rows += len(df)
        # 대여소 이름을 정수 코드로 (집계 속도와 메모리 절약)
        for k in pd.unique(pd.concat([df["start_station_key"], df["end_station_key"]], ignore_index=True)):
            codes.setdefault(k, len(codes))
        df["s"] = df["start_station_key"].map(codes).astype("int32")
        df["e"] = df["end_station_key"].map(codes).astype("int32")
        df["rt"] = df["is_round_trip"].astype(bool)
        df["model"] = df["bike_model"].astype(str)
        df["workday"] = (df["day_of_week"] < 5) & ~df["start_date"].isin(HOLIDAY_DATES)
        df["end_hour"] = df["end_time"].dt.hour.astype("int8")
        df["band"] = np.select(
            [df["workday"] & df["start_hour"].isin(AM_PEAK), df["workday"] & df["start_hour"].isin(PM_PEAK),
             df["workday"]],
            ["평일 오전 첨두(7~9시대)", "평일 오후 첨두(16~18시대)", "평일 기타"], default="주말·공휴일")
        df["dbin"] = np.clip(np.floor(df["duration_min"] / DUR_BIN), 0, 1440 / DUR_BIN).astype("int32")

        acc.add("daily", df.groupby(["start_date", "model", "rt"]).size())
        acc.add("hourly", df.groupby(["start_date", "start_hour"]).size())
        acc.add("st_start", df.groupby(["s", "workday", "start_hour"]).size())
        acc.add("st_end", df.groupby(["e", "workday", "end_hour"]).size())
        acc.add("st_year_start", df.groupby(["s", "year"]).size())
        acc.add("st_year_end", df.groupby(["e", "year"]).size())
        acc.add("od", df.groupby(["s", "e", "band"]).size())
        acc.add("dur", df.groupby(["year", "model", "workday", "start_hour", "rt", "dbin"]).size())
        del df
        gc.collect()
        if i % 10 == 0 or i == len(files):
            print(f"  집계 {i}/{len(files)}")

    names = pd.Series({v: k for k, v in codes.items()}).sort_index()
    return acc.store, names, total_rows


def build_tables(store, names):
    """누적 결과를 읽기 쉬운 테이블로 정리해 data/agg/에 저장한다."""
    t = {}
    daily = store["daily"].rename("trips").reset_index()
    daily.columns = ["date", "bike_model", "is_round_trip", "trips"]
    t["daily_detail"] = daily

    cal = pd.DataFrame({"date": pd.date_range(PERIOD_START, PERIOD_END, freq="D")})
    tot = daily.groupby("date")["trips"].sum()
    ebike = daily[daily["bike_model"].str.contains("EBIKE", case=False)].groupby("date")["trips"].sum()
    rtrip = daily[daily["is_round_trip"]].groupby("date")["trips"].sum()
    cal["trips"] = cal["date"].map(tot).fillna(0).astype("int64")
    cal["ebike_trips"] = cal["date"].map(ebike).fillna(0).astype("int64")
    cal["round_trips"] = cal["date"].map(rtrip).fillna(0).astype("int64")
    cal["year"], cal["month"] = cal["date"].dt.year, cal["date"].dt.month
    cal["day_of_week"] = cal["date"].dt.dayofweek
    cal["is_holiday"] = cal["date"].isin(HOLIDAY_DATES)
    cal["holiday_name"] = cal["date"].dt.strftime("%Y-%m-%d").map(BANK_HOLIDAYS).fillna("")
    cal["day_type"] = np.where(cal["is_holiday"], "공휴일", np.where(cal["day_of_week"] >= 5, "주말", "평일"))
    cal["workday"] = cal["day_type"] == "평일"
    cal["season"] = cal["month"].map(SEASON_OF)
    t["daily"] = cal

    hourly = store["hourly"].rename("trips").reset_index()
    hourly.columns = ["date", "hour", "trips"]
    t["hourly"] = hourly.merge(cal[["date", "day_of_week", "day_type", "workday"]], on="date", how="left")

    s_start = store["st_start"].rename("starts")
    s_start.index.names = ["code", "workday", "hour"]
    s_end = store["st_end"].rename("ends")
    s_end.index.names = ["code", "workday", "hour"]
    st = pd.concat([s_start, s_end], axis=1).fillna(0).astype("int64").reset_index()
    st.insert(0, "station_key", st["code"].map(names))
    t["station_hourly"] = st.drop(columns="code")

    y_start = store["st_year_start"].rename("starts")
    y_start.index.names = ["code", "year"]
    y_end = store["st_year_end"].rename("ends")
    y_end.index.names = ["code", "year"]
    sy = pd.concat([y_start, y_end], axis=1).fillna(0).astype("int64").reset_index()
    sy.insert(0, "station_key", sy["code"].map(names))
    t["station_year"] = sy.drop(columns="code")

    od = store["od"].rename("trips").reset_index()
    od.columns = ["s", "e", "band", "trips"]
    od.insert(0, "start_station_key", od["s"].map(names))
    od.insert(1, "end_station_key", od["e"].map(names))
    t["od"] = od.drop(columns=["s", "e"])

    dur = store["dur"].rename("trips").reset_index()
    dur.columns = ["year", "bike_model", "workday", "start_hour", "is_round_trip", "dbin", "trips"]
    dur["minutes"] = dur["dbin"] * DUR_BIN  # 구간 하한(분)
    t["duration_hist"] = dur

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in t.items():
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), AGG_DIR / f"{name}.parquet",
                       compression="zstd")
    return t


# ---------------------------------------------------------------------------
# 2) 대여 시간 분위수 (히스토그램에서 계산, 정밀도 0.1분)
# ---------------------------------------------------------------------------
def hist_quantiles(h: pd.DataFrame, qs=(0.25, 0.5, 0.75)):
    g = h.groupby("dbin")["trips"].sum().sort_index()
    cum = g.cumsum() / g.sum()
    out = {}
    for q in qs:
        b = cum.index[np.searchsorted(cum.values, q)]
        out[q] = (b + 0.5) * DUR_BIN  # 구간 중앙값
    return out


def dur_table(dur: pd.DataFrame, by):
    rows = []
    for key, grp in dur.groupby(by):
        q = hist_quantiles(grp)
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(by, key)) | {"대여 건수": int(grp["trips"].sum()), "25%(분)": round(q[0.25], 1),
                                          "중앙값(분)": round(q[0.5], 1), "75%(분)": round(q[0.75], 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3) 그래프
# ---------------------------------------------------------------------------
def make_figures(t):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = {}
    cal, hourly, dur = t["daily"], t["hourly"], t["duration_hist"]

    # F1. 일별 대여 건수
    fig, ax = new_ax(10, 4.2)
    roll = cal.set_index("date")["trips"].rolling(28, center=True, min_periods=14).mean()
    ax.plot(cal["date"], cal["trips"], color=C["s1_light"], linewidth=0.6, label="일별 대여 건수")
    ax.plot(roll.index, roll.values, color=C["s1"], linewidth=2, label="28일 이동평균")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C["ink2"])
    figs["daily"] = finish(fig, ax, "일별 대여 건수 (2023~2025)",
                           "옅은 선: 하루 단위 / 진한 선: 28일 이동평균", "f1_daily_trips.png")

    # F2. 월별 대여 건수 (연도별)
    fig, ax = new_ax()
    monthly = cal.groupby(["year", "month"])["trips"].sum().unstack(0)
    colors = [C["s1"], C["s2"], C["s3"]]
    for col, color in zip(monthly.columns, colors):
        ax.plot(monthly.index, monthly[col], color=color, linewidth=2, label=str(col))
    ax.set_xticks(range(1, 13), [f"{m}월" for m in range(1, 13)])
    ax.set_xlim(0.7, 13.2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    end_labels(ax, [(12, monthly[col].iloc[-1], str(col), c) for col, c in zip(monthly.columns, colors)])
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C["ink2"])
    figs["monthly"] = finish(fig, ax, "월별 대여 건수: 연도 비교", "같은 달끼리 연도별로 비교", "f2_monthly_by_year.png")

    # F3. 시간대별 하루 평균 대여 건수 (근무일 vs 비근무일)
    fig, ax = new_ax()
    n_days = cal.groupby("workday")["date"].nunique()
    prof = hourly.groupby(["workday", "hour"])["trips"].sum().unstack(0).div(n_days, axis=1)
    series = [(True, C["s1"], "평일"), (False, C["s2"], "주말·공휴일")]
    for wd, color, label in series:
        ax.plot(prof.index, prof[wd], color=color, linewidth=2, label=label)
    ax.set_xticks(range(0, 24, 2), [f"{h}시" for h in range(0, 24, 2)])
    ax.set_xlim(-0.5, 26)
    ax.set_ylim(bottom=0)
    end_labels(ax, [(23, prof[wd].iloc[-1], label, color) for wd, color, label in series])
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C["ink2"])
    figs["hourly"] = finish(fig, ax, "시간대별 하루 평균 대여 건수", "대여 시작 시각 기준, 날짜 유형별 하루 평균",
                            "f3_hourly_profile.png", ylabel="대여 건수 / 일")

    # F4. 요일 × 시간대 히트맵
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)
    fig.patch.set_facecolor(C["surface"])
    nh = hourly[hourly["day_type"] != "공휴일"]
    days_per_dow = cal[cal["day_type"] != "공휴일"].groupby("day_of_week")["date"].nunique()
    heat = nh.groupby(["day_of_week", "hour"])["trips"].sum().unstack().div(days_per_dow, axis=0)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    im = ax.imshow(heat.values, aspect="auto", cmap=cmap)
    ax.set_yticks(range(7), [f"{d}요일" for d in WEEKDAY_KO])
    ax.set_xticks(range(0, 24, 2), [f"{h}시" for h in range(0, 24, 2)])
    ax.tick_params(colors=C["ink2"], labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("하루 평균 대여 건수", color=C["ink2"], fontsize=9)
    cb.ax.tick_params(colors=C["ink2"], labelsize=8)
    cb.outline.set_visible(False)
    figs["heatmap"] = finish(fig, ax, "요일 × 시간대 이용 패턴", "공휴일 제외, 색이 진할수록 많음",
                             "f4_heatmap_weekday_hour.png")

    # F5. 1회 대여 시간 분포
    fig, ax = new_ax()
    h = dur.groupby((dur["minutes"]).astype(int))["trips"].sum()
    total = h.sum()
    shown = h[h.index < 60]
    ax.bar(shown.index + 0.5, shown.values / total * 100, width=0.9, color=C["s1"])
    med = hist_quantiles(dur)[0.5]
    ax.axvline(med, color=C["ink2"], linewidth=1, linestyle="--")
    ax.annotate(f"중앙값 {med:.1f}분", (med, ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=C["ink2"])
    over = h[h.index >= 60].sum() / total * 100
    figs["duration"] = finish(fig, ax, "1회 대여 시간 분포", f"1분 단위, 60분 이상({over:.1f}%)은 그래프에서 생략",
                              "f5_duration_hist.png", xlabel="대여 시간(분)", ylabel="전체 대여 중 비율(%)")

    # F6. 시간대별 대여 시간 중앙값
    fig, ax = new_ax()
    ends = []
    for wd, color, label in [(True, C["s1"], "평일"), (False, C["s2"], "주말·공휴일")]:
        sub = dur[dur["workday"] == wd]
        med_by_h = pd.Series({hr: hist_quantiles(g)[0.5] for hr, g in sub.groupby("start_hour")})
        ax.plot(med_by_h.index, med_by_h.values, color=color, linewidth=2, label=label)
        ends.append((med_by_h.index[-1], med_by_h.iloc[-1], label, color))
    ax.set_xticks(range(0, 24, 2), [f"{hr}시" for hr in range(0, 24, 2)])
    ax.set_xlim(-0.5, 26)
    ax.set_ylim(bottom=0)
    end_labels(ax, ends)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=C["ink2"])
    figs["duration_hour"] = finish(fig, ax, "시간대별 1회 대여 시간 중앙값", "대여 시작 시각 기준",
                                   "f6_duration_by_hour.png", ylabel="중앙값(분)")

    # F7. 월별 전기자전거 비중
    fig, ax = new_ax()
    m = cal.groupby(cal["date"].dt.to_period("M"))[["ebike_trips", "trips"]].sum()
    share = m["ebike_trips"] / m["trips"] * 100
    x = share.index.to_timestamp()
    ax.plot(x, share.values, color=C["s1"], linewidth=2)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    figs["ebike"] = finish(fig, ax, "월별 전기자전거 대여 비중", "전체 대여 중 PBSC_EBIKE 비율", "f7_ebike_share.png")

    # F8. 출발 대여 건수 상위 15개 대여소
    fig, ax = new_ax(9, 5)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    st = t["station_hourly"].groupby("station_key")["starts"].sum().sort_values(ascending=False)
    top = st.head(15)[::-1]
    ax.barh(range(len(top)), top.values, color=C["s1"], height=0.7)
    ax.set_yticks(range(len(top)), top.index, fontsize=8)
    for i, v in enumerate(top.values):
        ax.annotate(fmt_int(v), (v, i), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=C["ink2"])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    figs["top_stations"] = finish(fig, ax, "출발 대여 건수 상위 15개 대여소", "2023~2025년 합계",
                                  "f8_top_stations.png")

    # F9. 평일 오전 첨두 순유입(도착-출발) 상위·하위 대여소
    fig, ax = new_ax(9, 5.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    am = t["station_hourly"]
    n_wd = cal["workday"].sum()
    s_am = am[am["workday"] & am["hour"].isin(AM_PEAK)].groupby("station_key")["starts"].sum()
    e_am = am[am["workday"] & am["hour"].isin(AM_PEAK)].groupby("station_key")["ends"].sum()
    net = (e_am.reindex(s_am.index.union(e_am.index), fill_value=0)
           - s_am.reindex(s_am.index.union(e_am.index), fill_value=0)) / n_wd
    sel = pd.concat([net.nsmallest(10), net.nlargest(10)[::-1]]).sort_values()
    colors = [C["s1"] if v > 0 else C["neg"] for v in sel.values]
    ax.barh(range(len(sel)), sel.values, color=colors, height=0.7)
    ax.axvline(0, color=C["axis"], linewidth=1)
    ax.set_yticks(range(len(sel)), sel.index, fontsize=8)
    for i, v in enumerate(sel.values):
        ax.annotate(f"{v:+.0f}", (v, i), xytext=(4 if v > 0 else -4, 0), textcoords="offset points",
                    va="center", ha="left" if v > 0 else "right", fontsize=8, color=C["ink2"])
    figs["am_net"] = finish(fig, ax, "평일 오전 첨두(7~9시대) 순유입 상위·하위 10개 대여소",
                            "파랑: 도착이 출발보다 많음 / 빨강: 출발이 도착보다 많음 (평일 하루 평균)",
                            "f9_am_peak_net_flow.png", xlabel="하루 평균 (도착 - 출발)")
    return figs


# ---------------------------------------------------------------------------
# 4) 보고서 (관찰 수치만)
# ---------------------------------------------------------------------------
def write_report(t, figs, total_rows):
    cal, hourly, dur, st, sy, od = (t["daily"], t["hourly"], t["duration_hist"], t["station_hourly"],
                                    t["station_year"], t["od"])
    L = ["# 탐색적 데이터 분석(EDA) 결과", "",
         f"- 실행 일시: {datetime.now():%Y-%m-%d %H:%M}",
         f"- 입력: `data/processed/trips/` ({total_rows:,}행), 집계 테이블: `data/agg/`",
         "- 이 보고서는 **관찰(수치)**만 기록한다. 해석은 그래프와 표를 보고 별도로 작성한다.",
         "- 날짜 구분: 평일(월~금, 공휴일 제외) / 주말 / 공휴일(잉글랜드·웨일스 bank holiday, "
         "[GOV.UK](https://www.gov.uk/bank-holidays))", ""]

    # 0. 검산
    sums = {"원본 행 수": total_rows, "일별 집계 합계": int(cal["trips"].sum()),
            "시간대 집계 합계": int(hourly["trips"].sum()), "대여소 출발 합계": int(st["starts"].sum()),
            "대여소 도착 합계": int(st["ends"].sum()), "출발-도착(OD) 합계": int(od["trips"].sum()),
            "대여 시간 히스토그램 합계": int(dur["trips"].sum())}
    ok = len(set(sums.values())) == 1
    L += ["## 0. 집계 검산", "", "모든 집계의 합계가 원본 행 수와 같아야 한다.", "",
          to_md(pd.DataFrame({"항목": sums.keys(), "건수": [fmt_int(v) for v in sums.values()]})),
          "", f"- 결과: **{'모두 일치' if ok else '불일치 — 확인 필요'}**", ""]
    zero_days = cal.loc[cal["trips"] == 0, "date"].dt.strftime("%Y-%m-%d").tolist()
    L += [f"- 대여가 0건인 날: {len(zero_days)}일 {zero_days[:10]}", ""]

    # 1. 연도별 규모
    yr = cal.groupby("year").agg(대여건수=("trips", "sum"), 일수=("date", "size"))
    yr["하루 평균"] = (yr["대여건수"] / yr["일수"]).round(0)
    yr["전년 대비"] = (yr["대여건수"].pct_change() * 100).round(1).map(lambda v: "" if pd.isna(v) else f"{v:+.1f}%")
    yr["대여건수"] = yr["대여건수"].map(fmt_int)
    yr["하루 평균"] = yr["하루 평균"].map(fmt_int)
    L += ["## 1. 연도별 규모", "", to_md(yr.reset_index().rename(columns={"year": "연도"})), ""]

    # 2. 일별 추이
    L += ["## 2. 일별 추이", "", f"![일별 대여 건수]({figs['daily']})", ""]
    dt = cal.groupby(["year", "day_type"])["trips"].mean().round(0).unstack().reindex(columns=["평일", "주말", "공휴일"])
    L += ["날짜 유형별 하루 평균 대여 건수:", "", to_md(dt.map(fmt_int).reset_index().rename(columns={"year": "연도"})), ""]
    cols = ["date", "요일", "day_type", "holiday_name", "trips", "전기자전거 비중"]
    c2 = cal.copy()
    c2["요일"] = c2["day_of_week"].map(lambda d: WEEKDAY_KO[d])
    c2["전기자전거 비중"] = (c2["ebike_trips"] / c2["trips"].where(c2["trips"] > 0) * 100).round(1).map(
        lambda v: "" if pd.isna(v) else f"{v}%")
    c2["date"] = c2["date"].dt.strftime("%Y-%m-%d")
    top = c2.nlargest(15, "trips")[cols]
    bottom = c2.nsmallest(15, "trips")[cols]
    for df_ in (top, bottom):
        df_["trips"] = df_["trips"].map(fmt_int)
    L += ["대여가 가장 많았던 15일 (특별한 날인지는 외부 자료로 확인 필요):", "",
          to_md(top.rename(columns={"date": "날짜", "day_type": "유형", "holiday_name": "공휴일명", "trips": "대여 건수"})),
          "", "대여가 가장 적었던 15일:", "",
          to_md(bottom.rename(columns={"date": "날짜", "day_type": "유형", "holiday_name": "공휴일명", "trips": "대여 건수"})),
          ""]

    # 3. 월별·계절
    L += ["## 3. 월별·계절", "", f"![월별 대여 건수]({figs['monthly']})", ""]
    mon = cal.groupby(["year", "month"])["trips"].sum().unstack(0)
    mon.index = [f"{m}월" for m in mon.index]
    L += ["월별 대여 건수:", "", to_md(mon.map(fmt_int), index=True), ""]
    sea = cal.groupby(["season", "year"])["trips"].mean().round(0).unstack().reindex(["봄", "여름", "가을", "겨울"])
    L += ["계절별 하루 평균 대여 건수 (겨울 = 12·1·2월, 같은 해의 1·2·12월로 계산):", "",
          to_md(sea.map(fmt_int), index=True), ""]

    # 4. 시간대·요일
    n_days = cal.groupby("workday")["date"].nunique()
    prof = hourly.groupby(["workday", "hour"])["trips"].sum().unstack(0).div(n_days, axis=1).round(0)
    prof.columns = ["주말·공휴일", "평일"] if list(prof.columns) == [False, True] else prof.columns
    L += ["## 4. 시간대·요일", "", f"![시간대별 하루 평균]({figs['hourly']})", "",
          f"![요일 × 시간대]({figs['heatmap']})", ""]
    for col in ["평일", "주말·공휴일"]:
        top3 = prof[col].nlargest(3)
        L.append(f"- {col} 상위 3개 시간대: " + ", ".join(f"{h}시대({fmt_int(v)}건/일)" for h, v in top3.items()))
    wd_day = hourly[hourly["workday"]].groupby("hour")["trips"].sum()
    am_share = wd_day[list(AM_PEAK)].sum() / wd_day.sum() * 100
    pm_share = wd_day[list(PM_PEAK)].sum() / wd_day.sum() * 100
    L += [f"- 평일 대여 중 7~9시대 비중: {am_share:.1f}%, 16~18시대 비중: {pm_share:.1f}%", ""]
    dow = cal[cal["day_type"] != "공휴일"].groupby("day_of_week")["trips"].mean().round(0)
    dow.index = [f"{WEEKDAY_KO[d]}요일" for d in dow.index]
    L += ["요일별 하루 평균 대여 건수 (공휴일 제외):", "", to_md(dow.map(fmt_int).to_frame("하루 평균"), index=True), ""]
    L += ["시간대별 하루 평균 대여 건수:", "", to_md(prof.map(fmt_int), index=True), ""]

    # 5. 대여 시간
    L += ["## 5. 1회 대여 시간", "", f"![대여 시간 분포]({figs['duration']})", "",
          f"![시간대별 대여 시간]({figs['duration_hour']})", ""]
    q = hist_quantiles(dur, (0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
    L += ["전체 분위수(분, 0.1분 구간 중앙값 기준): " + ", ".join(f"{int(k * 100)}% {v:.1f}" for k, v in q.items()), ""]
    d2 = dur.copy()
    d2["날짜 유형"] = np.where(d2["workday"], "평일", "주말·공휴일")
    d2["왕복 여부"] = np.where(d2["is_round_trip"], "왕복", "편도")
    L += ["날짜 유형별:", "", to_md(dur_table(d2, ["날짜 유형"])), "",
          "왕복(출발 = 도착 대여소) 여부별:", "", to_md(dur_table(d2, ["왕복 여부"])), "",
          "자전거 모델 × 연도별 (모델 비교는 연도를 나눠서 볼 것):", "",
          to_md(dur_table(d2.rename(columns={"bike_model": "모델", "year": "연도"}), ["연도", "모델"])), ""]

    # 6. 자전거 모델
    eb = cal.groupby("year")[["ebike_trips", "trips"]].sum()
    eb["전기자전거 비중"] = (eb["ebike_trips"] / eb["trips"] * 100).round(1).astype(str) + "%"
    L += ["## 6. 자전거 모델", "", f"![전기자전거 비중]({figs['ebike']})", "",
          to_md(eb[["전기자전거 비중"]].reset_index().rename(columns={"year": "연도"})), ""]

    # 7. 대여소
    tot_st = st.groupby("station_key")[["starts", "ends"]].sum()
    active = sy[sy["starts"] + sy["ends"] > 0].groupby("year")["station_key"].nunique()
    keys_by_year = {y: set(g.loc[g["starts"] + g["ends"] > 0, "station_key"]) for y, g in sy.groupby("year")}
    yrs = sorted(keys_by_year)
    L += ["## 7. 대여소", "",
          f"- 분석 기간에 이용된 대여소: {len(tot_st):,}개",
          "- 연도별 이용된 대여소 수: " + ", ".join(f"{y}년 {n:,}개" for y, n in active.items())]
    for a, b in zip(yrs, yrs[1:]):
        L.append(f"  - {a}→{b}: 새로 나타난 대여소 {len(keys_by_year[b] - keys_by_year[a]):,}개, "
                 f"사라진 대여소 {len(keys_by_year[a] - keys_by_year[b]):,}개")
    s_sorted = tot_st["starts"].sort_values(ascending=False)
    k10 = max(1, int(round(len(s_sorted) * 0.1)))
    L += [f"- 출발 기준 상위 10% 대여소({k10}개)가 전체 출발의 {s_sorted.head(k10).sum() / s_sorted.sum() * 100:.1f}%를 차지",
          f"- 대여소별 출발 건수: 중앙값 {s_sorted.median():,.0f}건, 최소 {s_sorted.min():,}건, 최대 {s_sorted.max():,}건",
          "", f"![상위 대여소]({figs['top_stations']})", "", f"![오전 첨두 순유입]({figs['am_net']})", ""]
    n_wd = int(cal["workday"].sum())
    for band, label in [(AM_PEAK, "오전 첨두(7~9시대)"), (PM_PEAK, "오후 첨두(16~18시대)")]:
        sub = st[st["workday"] & st["hour"].isin(band)].groupby("station_key")[["starts", "ends"]].sum()
        sub["순유입(도착-출발)/일"] = ((sub["ends"] - sub["starts"]) / n_wd).round(1)
        L += [f"평일 {label} 순유입 상위 5 / 하위 5:", "",
              to_md(pd.concat([sub.nlargest(5, "순유입(도착-출발)/일"), sub.nsmallest(5, "순유입(도착-출발)/일")])
                    [["순유입(도착-출발)/일"]], index=True), ""]

    # 8. 출발-도착 흐름
    od_tot = od.groupby(["start_station_key", "end_station_key"])["trips"].sum().reset_index()
    rt_share = od_tot.loc[od_tot["start_station_key"] == od_tot["end_station_key"], "trips"].sum() / od_tot["trips"].sum() * 100
    one_way = od_tot[od_tot["start_station_key"] != od_tot["end_station_key"]].nlargest(10, "trips")
    one_way["trips"] = one_way["trips"].map(fmt_int)
    L += ["## 8. 출발-도착 흐름", "",
          f"- 서로 다른 출발-도착 대여소 쌍: {(od_tot['start_station_key'] != od_tot['end_station_key']).sum():,}개",
          f"- 왕복(출발 = 도착 대여소) 비중: {rt_share:.2f}%",
          "", "이용이 가장 많은 편도 흐름 10개:", "",
          to_md(one_way.rename(columns={"start_station_key": "출발", "end_station_key": "도착", "trips": "대여 건수"})), ""]
    L += ["## 9. 해석 작성 시 확인할 것", "",
          "- 대여 상위·하위 날짜가 특별한 날(공휴일, 파업, 행사, 날씨)인지 외부 자료로 확인하고 출처를 적는다.",
          "- 연도 간 차이는 대여소 수 변화, 전기자전거 확대 등 공급 측 변화와 함께 봐야 한다 (원인 단정 금지).",
          "- 대여 시간은 반납 시각이 아니라 대여 시작 시각 기준으로 묶었다.", ""]
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    setup_font()
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": list(BANK_HOLIDAYS), "name": list(BANK_HOLIDAYS.values()),
                  "source": "https://www.gov.uk/bank-holidays (2026-11-16 확인)"}).to_csv(
        EXT_DIR / "uk_bank_holidays_2023_2025.csv", index=False, encoding="utf-8-sig")

    print("[1/3] 파일별 집계")
    store, names, total_rows = aggregate()
    t = build_tables(store, names)
    print("[2/3] 그래프")
    figs = make_figures(t)
    print("[3/3] 보고서")
    write_report(t, figs, total_rows)
    print(f"[완료] {REPORT_PATH}\n       {FIG_DIR}\n       {AGG_DIR}")


if __name__ == "__main__":
    main()