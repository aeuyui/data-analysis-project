"""
3차 분석 스크립트(04~07)가 함께 쓰는 설정과 함수

- 경로, 공휴일, 운영 장애일
- 그래프 스타일 (03_eda.py와 같은 팔레트)
- 대여소 시간대 프로파일 계산 (질문 1·3에서 같은 방식으로 쓰기 위해 한곳에 둠)

이 파일은 proj 폴더 바로 아래에 두며, 직접 실행하지 않는다.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
TRIPS_DIR = PROCESSED_DIR / "trips"
AGG_DIR = PROJECT_DIR / "data" / "agg"
OUT_DIR = PROJECT_DIR / "03_analysis-modeling"
FIG_DIR = OUT_DIR / "figures"

# ---------------------------------------------------------------------------
# 날짜 구분
# ---------------------------------------------------------------------------
# 잉글랜드·웨일스 bank holiday (GOV.UK, https://www.gov.uk/bank-holidays, 2026-11-16 확인)
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

# 운영 장애일: 수요가 아니라 공급이 끊긴 날이라 분석에서 제외 (2차 문서 8.1절)
OUTAGE_DAYS = {"2025-08-05": "소프트웨어 업그레이드 지연으로 앱·대여소 대여 불가 (184건)"}
OUTAGE_DATES = pd.to_datetime(list(OUTAGE_DAYS))

PERIOD_START, PERIOD_END = "2023-01-01", "2025-12-31"
AM_PEAK, PM_PEAK = list(range(7, 10)), list(range(16, 19))   # 7~9시대, 16~18시대
WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]


def day_counts() -> pd.DataFrame:
    """연도 × 근무일 여부별 날짜 수 (운영 장애일 제외). 하루 평균을 낼 때 분모로 쓴다."""
    cal = pd.DataFrame({"date": pd.date_range(PERIOD_START, PERIOD_END, freq="D")})
    cal = cal[~cal["date"].isin(OUTAGE_DATES)]
    cal["year"] = cal["date"].dt.year
    cal["workday"] = (cal["date"].dt.dayofweek < 5) & ~cal["date"].isin(HOLIDAY_DATES)
    return cal.groupby(["year", "workday"]).size().rename("days").reset_index()


# ---------------------------------------------------------------------------
# 대여소 시간대 프로파일
# ---------------------------------------------------------------------------
def load_station_hourly(variant: str = "base") -> pd.DataFrame:
    sh = pd.read_parquet(AGG_DIR / "station_hourly_year.parquet")
    return sh[sh["variant"] == variant].drop(columns="variant")


def station_profiles(sh: pd.DataFrame, years=None, stations=None) -> pd.DataFrame:
    """
    대여소별 96차원 프로파일: (출발/도착) × (근무일/비근무일) × 24시간의 '비율'.
    - 각 대여소의 전체 대여(출발+도착) 합으로 나누므로 규모가 아니라 '모양'만 비교한다.
    - years: 사용할 연도 목록 (None이면 전체)
    """
    df = sh if years is None else sh[sh["year"].isin(list(years))]
    long = df.melt(id_vars=["station_key", "year", "workday", "hour"], value_vars=["starts", "ends"],
                   var_name="kind", value_name="trips")
    wide = long.pivot_table(index="station_key", columns=["kind", "workday", "hour"], values="trips",
                            aggfunc="sum", fill_value=0)
    full_cols = pd.MultiIndex.from_product([["starts", "ends"], [True, False], range(24)],
                                           names=["kind", "workday", "hour"])
    wide = wide.reindex(columns=full_cols, fill_value=0)
    if stations is not None:
        wide = wide.reindex(list(stations), fill_value=0)
    total = wide.sum(axis=1)
    return wide.div(total.where(total > 0), axis=0).fillna(0)


def station_metrics(sh: pd.DataFrame, years=None) -> pd.DataFrame:
    """해석용 지표: 하루 평균 대여, 비근무일/근무일 이용 비, 오전·오후 첨두 도착 비율."""
    df = sh if years is None else sh[sh["year"].isin(list(years))]
    dc = day_counts()
    if years is not None:
        dc = dc[dc["year"].isin(list(years))]
    days = dc.groupby("workday")["days"].sum()
    g = df.groupby(["station_key", "workday"])[["starts", "ends"]].sum()
    tot = g.sum(axis=1).unstack(fill_value=0).reindex(columns=[True, False], fill_value=0)
    out = pd.DataFrame(index=tot.index)
    out["trips_per_workday"] = tot[True] / days.get(True, 1)
    out["trips_per_nonworkday"] = tot[False] / days.get(False, 1)
    out["trips_per_day"] = (tot[True] + tot[False]) / days.sum()
    out["nonwork_rate"] = out["trips_per_nonworkday"] / out["trips_per_workday"].where(out["trips_per_workday"] > 0)
    for name, hours in [("am", AM_PEAK), ("pm", PM_PEAK)]:
        p = df[df["workday"] & df["hour"].isin(hours)].groupby("station_key")[["starts", "ends"]].sum()
        p = p.reindex(out.index, fill_value=0)
        out[f"{name}_arrival_ratio"] = p["ends"] / (p["starts"] + p["ends"]).where(lambda s: s > 0)
        out[f"{name}_net_per_workday"] = (p["ends"] - p["starts"]) / days.get(True, 1)
    return out


# ---------------------------------------------------------------------------
# 그래프 스타일 (dataviz 기본 팔레트, 라이트 모드)
# ---------------------------------------------------------------------------
C = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
    "s1_light": "#9ec5f4", "neg": "#e34948",
}


def setup_font():
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


def style_ax(ax, grid_axis="y"):
    ax.set_facecolor(C["surface"])
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(C["axis"])
    ax.tick_params(colors=C["ink2"], labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def new_ax(width=9, height=4.2, ncols=1, nrows=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), dpi=150, **kw)
    fig.patch.set_facecolor(C["surface"])
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        style_ax(ax)
    return fig, axes


def finish(fig, title, subtitle, filename, bottom_in=0.0):
    """제목·부제목을 인치 단위로 배치해 그림 높이와 상관없이 겹치지 않게 한다.
    bottom_in: 그림 아래에 범례 등을 둘 공간(인치)."""
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.10 / h, ha="left", va="top", fontsize=12, color=C["ink"],
                 fontweight="bold")
    if subtitle:
        fig.text(0.01, 1 - 0.40 / h, subtitle, ha="left", va="top", fontsize=9, color=C["ink2"])
    fig.tight_layout(rect=(0, bottom_in / h, 1, 1 - 0.65 / h))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / filename, facecolor=C["surface"])
    plt.close(fig)
    return f"figures/{filename}"


def to_md(df: pd.DataFrame, index=False) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


def fmt_int(x) -> str:
    return f"{x:,.0f}"
