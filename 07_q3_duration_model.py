"""
3차 4단계 — 질문 3: 1회 대여 시간은 시간대, 요일, 자전거 모델, 왕복 여부와 어떤 관련이 있는가?

두 가지로 답한다.
A. 관련성(설명): 시간대·장소 구성을 맞춘 뒤에도 전기자전거의 대여 시간이 더 긴가 (층화 비교)
B. 예측(검증): 대여 '시작 시점'에 알 수 있는 정보로 대여 시간을 얼마나 맞힐 수 있는가

예측 설계
- Target: log(대여 시간[분])  (꼬리가 길어 로그로 변환, 예측값은 다시 분으로 되돌려 평가)
- Feature (대여 시작 시점에 알 수 있는 것만):
    시작 시간대, 날짜 유형(평일/주말/공휴일), 월, 자전거 모델, 출발 대여소 유형(질문 1 방식), 출발 대여소 목표 인코딩
- Train / Test: 2023~2024년 학습, 2025년 평가 (시간 순서대로 나눠 '미래 예측' 상황을 재현)
- Baseline: (B0) 전체 중앙값, (B1) 시간대 × 평일 여부별 중앙값
- 모델: (M1) 선형 회귀(Ridge, 범주 원-핫), (M2) 그레이디언트 부스팅(HistGradientBoosting)
- 지표: MAE(분), 오차 중앙값(분), 로그 척도 MAE, 로그 척도 R²

Data Leakage 점검
1. 도착 대여소, 왕복 여부, 종료 시각은 대여가 끝나야 알 수 있으므로 특성에서 제외한다.
   → 넣었을 때 성능이 얼마나 '부풀려지는지' 별도 모델(M2+누수)로 보여 준다 (사용하지 않음).
2. 출발 대여소 목표 인코딩은 학습 데이터 안에서 5겹 교차(out-of-fold)로 계산한다.
   → 자기 자신의 대여 시간이 자신의 특성에 섞이지 않게 한다. 평가 데이터에는 학습 데이터의 값만 쓴다.
3. 출발 대여소 유형은 학습 연도(2023~2024)의 이용 자료만으로 다시 군집화한다.
4. 연도 자체는 특성에 넣지 않는다 (학습에 없는 2025년을 외삽하게 되므로).

입력: data/processed/q3_sample.parquet, data/agg/station_hourly_year.parquet, data/agg/station_clusters.csv(k 참고)
출력: data/agg/q3_metrics.csv, 03_analysis-modeling/q3_report.md, figures/q3_*.png

사용법 (proj 폴더에서 실행)
    python 07_q3_duration_model.py
"""

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from common import (AGG_DIR, C, OUT_DIR, PROCESSED_DIR, finish, fmt_int, load_station_hourly, new_ax, setup_font,
                    station_profiles, to_md)

TRAIN_YEARS, TEST_YEARS = [2023, 2024], [2025]
MIN_TRIPS_PER_YEAR = 1000
TE_SMOOTH, TE_FOLDS = 50, 5
IMPORTANCE_N = 50_000
SEED = 0

CAT = ["start_hour", "day_type", "month", "bike_model", "start_cluster"]
NUM = ["start_station_te"]
LEAK_CAT, LEAK_NUM = ["is_round_trip"], ["end_station_te"]
NAMES_KO = {"start_hour": "시작 시간대", "day_type": "날짜 유형", "month": "월", "bike_model": "자전거 모델",
            "start_cluster": "출발 대여소 유형", "start_station_te": "출발 대여소(목표 인코딩)",
            "is_round_trip": "왕복 여부 (누수)", "end_station_te": "도착 대여소 (누수)"}


# ---------------------------------------------------------------------------
# 특성 만들기
# ---------------------------------------------------------------------------
def train_station_clusters(k: int) -> pd.Series:
    """학습 연도 자료만으로 대여소 유형을 다시 만든다 (평가 연도 정보 누수 방지)."""
    sh = load_station_hourly("base")
    sh = sh[sh["year"].isin(TRAIN_YEARS)]
    tot = sh.groupby(["station_key", "year"])[["starts", "ends"]].sum().sum(axis=1).unstack(fill_value=0)
    eligible = tot.index[(tot >= MIN_TRIPS_PER_YEAR).all(axis=1)]
    X = station_profiles(sh, stations=eligible)
    km = KMeans(n_clusters=k, n_init=30, random_state=SEED).fit(X.to_numpy())
    return pd.Series(km.labels_, index=eligible)


def target_encode(train: pd.DataFrame, test: pd.DataFrame, col: str):
    """평활 평균 목표 인코딩. 학습 데이터는 out-of-fold, 평가 데이터는 학습 전체 통계."""
    prior = train["y"].mean()

    def enc(df):
        s = df.groupby(col, observed=True)["y"].agg(["sum", "count"])
        return (s["sum"] + TE_SMOOTH * prior) / (s["count"] + TE_SMOOTH)

    oof = np.empty(len(train))
    for tr_idx, va_idx in KFold(TE_FOLDS, shuffle=True, random_state=SEED).split(train):
        e = enc(train.iloc[tr_idx])
        oof[va_idx] = train.iloc[va_idx][col].map(e).astype(float).fillna(prior).to_numpy()
    full = enc(train)
    return oof, test[col].map(full).astype(float).fillna(prior).to_numpy()


def as_categories(train, test, cols):
    for c in cols:
        cats = pd.Index(sorted(train[c].astype(str).unique()))
        train[c] = pd.Categorical(train[c].astype(str), categories=cats)
        test[c] = pd.Categorical(test[c].astype(str), categories=cats)  # 학습에 없던 값은 결측으로


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def evaluate(y_log, pred_log) -> dict:
    actual, pred = np.exp(y_log), np.exp(pred_log)
    err = np.abs(pred - actual)
    return {"MAE(분)": err.mean(), "오차 중앙값(분)": np.median(err),
            "로그 MAE": np.abs(pred_log - y_log).mean(), "로그 R²": r2_score(y_log, pred_log)}


def make_hgb():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True,
                                         validation_fraction=0.1, random_state=SEED, categorical_features="from_dtype")


def main():
    setup_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED_DIR / "q3_sample.parquet")
    df = df[df["duration_min"] > 0].copy()
    df["y"] = np.log(df["duration_min"])
    df["is_round_trip"] = df["is_round_trip"].astype(bool)

    # 대여소 유형 (k는 질문 1 결과를 따름)
    cpath = AGG_DIR / "station_clusters.csv"
    k = int(pd.read_csv(cpath)["cluster"].max() + 1) if cpath.exists() else 4
    cl = train_station_clusters(k)
    df["start_cluster"] = df["start_station_key"].map(cl).fillna(-1).astype(int)  # -1 = 이용이 적은 대여소

    train = df[df["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    test = df[df["year"].isin(TEST_YEARS)].reset_index(drop=True)
    train["start_station_te"], test["start_station_te"] = target_encode(train, test, "start_station_key")
    train["end_station_te"], test["end_station_te"] = target_encode(train, test, "end_station_key")
    as_categories(train, test, CAT + LEAK_CAT)
    print(f"학습 {len(train):,}행 / 평가 {len(test):,}행, 대여소 유형 k = {k}")

    results, preds = [], {}

    # Baseline
    b0 = np.log(train["duration_min"].median())
    preds["B0"] = np.full(len(test), b0)
    med = train.groupby(["start_hour", "day_type"], observed=True)["duration_min"].median()
    key = pd.MultiIndex.from_arrays([test["start_hour"], test["day_type"]])
    preds["B1"] = np.log(med.reindex(key).fillna(train["duration_min"].median()).to_numpy())

    # M1: Ridge
    ridge = make_pipeline(ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                                             ("num", "passthrough", NUM)]), Ridge(alpha=1.0))
    ridge.fit(train[CAT + NUM].astype({c: str for c in CAT}), train["y"])
    preds["M1"] = ridge.predict(test[CAT + NUM].astype({c: str for c in CAT}))
    m1_train = evaluate(train["y"].to_numpy(), ridge.predict(train[CAT + NUM].astype({c: str for c in CAT})))

    # M2: Gradient boosting
    hgb = make_hgb().fit(train[CAT + NUM], train["y"])
    preds["M2"] = hgb.predict(test[CAT + NUM])
    m2_train = evaluate(train["y"].to_numpy(), hgb.predict(train[CAT + NUM]))

    # 누수 시연 (사용하지 않음)
    hgb_leak = make_hgb().fit(train[CAT + NUM + LEAK_CAT + LEAK_NUM], train["y"])
    preds["M2+누수"] = hgb_leak.predict(test[CAT + NUM + LEAK_CAT + LEAK_NUM])

    labels = {"B0": "Baseline 0: 전체 중앙값", "B1": "Baseline 1: 시간대×날짜유형 중앙값",
              "M1": "M1: 선형 회귀 (Ridge)", "M2": "M2: 그레이디언트 부스팅",
              "M2+누수": "참고: M2 + 도착 정보 (누수, 사용 불가)"}
    y_test = test["y"].to_numpy()
    for kname, p in preds.items():
        results.append({"모델": labels[kname], **evaluate(y_test, p)})
    res = pd.DataFrame(results)
    res.to_csv(AGG_DIR / "q3_metrics.csv", index=False, encoding="utf-8-sig")

    # 전처리 기준 민감도: 기준을 바꾼 학습·평가 데이터로 B1과 M2를 다시 비교
    # (목표 인코딩은 현재 기준 학습 데이터 값을 그대로 쓴다)
    def is_short_rt(d):
        return ((d["is_round_trip"].astype(str) == "True") & (d["duration_min"] <= 2)).to_numpy()

    sens_rows = []
    for desc, keep in [("현재 기준", None),
                       ("짧은 왕복 2분 이하 추가 제외", lambda d: ~is_short_rt(d)),
                       ("3시간 초과 추가 제외", lambda d: (d["duration_min"] <= 180).to_numpy())]:
        if keep is None:
            tr, te, p_b1, p_m2 = train, test, preds["B1"], preds["M2"]
        else:
            tr, te = train[keep(train)], test[keep(test)]
            med_v = tr.groupby(["start_hour", "day_type"], observed=True)["duration_min"].median()
            key_v = pd.MultiIndex.from_arrays([te["start_hour"], te["day_type"]])
            p_b1 = np.log(med_v.reindex(key_v).fillna(tr["duration_min"].median()).to_numpy())
            p_m2 = make_hgb().fit(tr[CAT + NUM], tr["y"]).predict(te[CAT + NUM])
        e_b1 = evaluate(te["y"].to_numpy(), p_b1)["MAE(분)"]
        e_m2 = evaluate(te["y"].to_numpy(), p_m2)["MAE(분)"]
        sens_rows.append({"기준": desc, "평가 행 수": fmt_int(len(te)), "B1 MAE(분)": round(e_b1, 3),
                          "M2 MAE(분)": round(e_m2, 3), "B1 대비 변화": f"{(e_m2 - e_b1) / e_b1 * 100:+.1f}%"})

    # 오류 분석 (M2, 평가 데이터)
    ea = test[["bike_model", "day_type", "is_round_trip", "duration_min"]].copy()
    ea["pred_M2"] = np.exp(preds["M2"])
    ea["pred_B1"] = np.exp(preds["B1"])
    ea["err_M2"] = (ea["pred_M2"] - ea["duration_min"]).abs()
    ea["err_B1"] = (ea["pred_B1"] - ea["duration_min"]).abs()
    ea["bias_M2"] = ea["pred_M2"] - ea["duration_min"]
    bins = [0, 10, 20, 40, 60, np.inf]
    blabels = ["10분 미만", "10~20분", "20~40분", "40~60분", "60분 이상"]
    ea["실제 구간"] = pd.cut(ea["duration_min"], bins=bins, labels=blabels, right=False)

    def err_by(col):
        g = ea.groupby(col, observed=True).agg(건수=("err_M2", "size"), MAE_M2=("err_M2", "mean"),
                                               MAE_B1=("err_B1", "mean"), 평균_편향_M2=("bias_M2", "mean"))
        g["건수"] = g["건수"].map(fmt_int)
        return g.round(2)

    # 특성 중요도 (평가 데이터 일부, 섞었을 때 로그 MAE가 얼마나 나빠지는가)
    smp = test.sample(min(IMPORTANCE_N, len(test)), random_state=SEED)
    pi = permutation_importance(hgb, smp[CAT + NUM], smp["y"], scoring="neg_mean_absolute_error",
                                n_repeats=3, random_state=SEED)
    imp = pd.DataFrame({"특성": [NAMES_KO[c] for c in CAT + NUM], "로그 MAE 증가": pi.importances_mean,
                        "표준편차": pi.importances_std}).sort_values("로그 MAE 증가", ascending=False)

    # A. 층화 비교: 같은 연도·시간대·날짜 유형·출발 대여소 유형 안에서 전기자전거 − 일반 자전거 중앙값 차이
    strata = ["year", "start_hour", "day_type", "start_cluster"]
    allx = df.copy()
    g = allx.groupby(strata + ["bike_model"], observed=True)["duration_min"].agg(["median", "size"]).unstack("bike_model")
    models = list(g["median"].columns)
    strat_rows = []
    if len(models) == 2:
        ebike = [m for m in models if "EBIKE" in str(m).upper()]
        classic = [m for m in models if m not in ebike]
        if ebike and classic:
            e, c = ebike[0], classic[0]
            g = g.dropna()
            g = g[(g[("size", e)] >= 20) & (g[("size", c)] >= 20)]
            g["diff"] = g[("median", e)] - g[("median", c)]
            g["w"] = g[[("size", e), ("size", c)]].min(axis=1)
            for y, gy in g.groupby(level="year"):
                raw = allx[allx["year"] == y].groupby("bike_model", observed=True)["duration_min"].median()
                strat_rows.append({"연도": y, "단순 중앙값 차이(분)": round(raw[e] - raw[c], 2),
                                   "층화 후 차이(가중 평균, 분)": round(np.average(gy["diff"], weights=gy["w"]), 2),
                                   "전기자전거가 더 긴 층 비율": f"{(gy['diff'] > 0).mean() * 100:.0f}%",
                                   "비교한 층 수": len(gy)})

    # 그래프 1: 모델 비교
    fig, ax = new_ax(9, 3.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    r = res.iloc[::-1]
    colors = [C["muted"] if "누수" in m else C["s1"] for m in r["모델"]]
    ax.barh(range(len(r)), r["MAE(분)"], color=colors, height=0.6)
    ax.set_yticks(range(len(r)), r["모델"], fontsize=9)
    for i, v in enumerate(r["MAE(분)"]):
        ax.annotate(f"{v:.2f}분", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                    color=C["ink2"])
    ax.set_xlabel("평가 데이터(2025년) MAE (분, 낮을수록 좋음)", color=C["ink2"], fontsize=9)
    f_cmp = finish(fig, "대여 시간 예측: Baseline과 모델 비교", "2023~2024년으로 학습, 2025년으로 평가 / 회색은 누수 시연(사용 불가)",
                   "q3_model_comparison.png")

    # 그래프 2: 특성 중요도
    fig, ax = new_ax(8, 3.6)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ii = imp.iloc[::-1]
    ax.barh(range(len(ii)), ii["로그 MAE 증가"], xerr=ii["표준편차"], color=C["s1"], height=0.6,
            error_kw={"ecolor": C["ink2"], "elinewidth": 1})
    ax.set_yticks(range(len(ii)), ii["특성"], fontsize=9)
    ax.set_xlabel("값을 무작위로 섞었을 때 로그 MAE 증가량", color=C["ink2"], fontsize=9)
    f_imp = finish(fig, "M2 특성 중요도 (순열 중요도)", f"2025년 평가 데이터 {len(smp):,}행, 3회 반복 평균 ± 표준편차",
                   "q3_importance.png")

    # 그래프 3: 실제 대여 시간 구간별 오차
    eb = err_by("실제 구간")
    fig, ax = new_ax(8.5, 3.8)
    x = np.arange(len(eb))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, eb["MAE_B1"], width=w, color=C["s2"], label="Baseline 1")
    ax.bar(x + w / 2 + 0.01, eb["MAE_M2"], width=w, color=C["s1"], label="M2")
    ax.set_xticks(x, eb.index)
    ax.set_ylabel("MAE (분)", color=C["ink2"], fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=C["ink2"])
    f_err = finish(fig, "실제 대여 시간 구간별 오차", "긴 대여일수록 시작 시점 정보만으로는 맞히기 어렵다", "q3_error_by_duration.png")

    # 보고서
    fmt = res.copy()
    for col in ["MAE(분)", "오차 중앙값(분)", "로그 MAE"]:
        fmt[col] = fmt[col].round(3)
    fmt["로그 R²"] = fmt["로그 R²"].round(3) + 0.0  # -0.0 표시 방지
    base_mae = res.loc[res["모델"] == labels["B1"], "MAE(분)"].iloc[0]
    m2_mae = res.loc[res["모델"] == labels["M2"], "MAE(분)"].iloc[0]
    leak_mae = res.loc[res["모델"] == labels["M2+누수"], "MAE(분)"].iloc[0]
    over = pd.DataFrame([{"모델": "M1 (Ridge)", **{f"학습 {k_}": round(v, 3) for k_, v in m1_train.items()},
                          **{f"평가 {k_}": round(v, 3) for k_, v in evaluate(y_test, preds['M1']).items()}},
                         {"모델": "M2 (부스팅)", **{f"학습 {k_}": round(v, 3) for k_, v in m2_train.items()},
                          **{f"평가 {k_}": round(v, 3) for k_, v in evaluate(y_test, preds['M2']).items()}}])
    imp_fmt = imp.assign(**{"로그 MAE 증가": imp["로그 MAE 증가"].round(4), "표준편차": imp["표준편차"].round(4)})

    L = ["# 질문 3. 1회 대여 시간", "",
         f"- 실행 일시: {datetime.now():%Y-%m-%d %H:%M}",
         "- 질문: 1회 대여 시간은 시간대, 요일, 자전거 모델, 왕복 여부와 어떤 관련이 있는가?",
         "- 데이터: `q3_sample.parquet` (분석 데이터의 약 5% 표본, trip_id 기준 추출, 운영 장애일 제외)",
         "- 이 보고서는 **관찰(수치)**을 기록한다.", "",
         "## 1. 설계", "",
         "| 항목 | 내용 |", "|---|---|",
         "| Target | log(대여 시간[분]) — 오른쪽 꼬리가 길어 로그 변환, 평가는 분으로 되돌려 계산 |",
         f"| Train / Test | {'·'.join(map(str, TRAIN_YEARS))}년 {len(train):,}행 / {'·'.join(map(str, TEST_YEARS))}년 {len(test):,}행 (시간 순서 분할) |",
         "| Feature | " + ", ".join(NAMES_KO[c] for c in CAT + NUM) + " |",
         f"| 출발 대여소 유형 | 학습 연도 자료만으로 K-means 재군집화 (k = {k}, 질문 1과 같은 k). 이용이 적은 대여소는 '-1' |",
         "| 제외한 정보 | 도착 대여소, 왕복 여부, 종료 시각(대여가 끝나야 알 수 있음), 연도(학습에 없는 해를 외삽) |",
         f"| 목표 인코딩 | 출발 대여소별 로그 대여 시간 평활 평균 (평활 {TE_SMOOTH}), 학습 데이터는 {TE_FOLDS}겹 out-of-fold |", "",
         "## 2. 결과", "", f"![모델 비교]({f_cmp})", "", to_md(fmt), "",
         f"- 가장 강한 Baseline(B1) 대비 M2의 MAE 변화: {base_mae:.2f}분 → {m2_mae:.2f}분 "
         f"({(m2_mae - base_mae) / base_mae * 100:+.1f}%)",
         f"- 누수 시연: 도착 정보(도착 대여소, 왕복 여부)를 넣으면 MAE {leak_mae:.2f}분 "
         f"({(leak_mae - m2_mae) / m2_mae * 100:+.1f}% vs M2). "
         + ("대여 시작 시점에는 알 수 없는 정보로 얻은 개선이므로, 이 정보를 넣은 평가는 실제 성능을 부풀린다."
            if leak_mae < m2_mae * 0.99 else
            "이 데이터에서는 도착 정보를 넣어도 개선이 작았다. 그래도 대여 시작 시점에 알 수 없는 정보이므로 쓰지 않는다."), "",
         "### 학습 vs 평가 (과적합 확인)", "", to_md(over), "",
         "### 전처리 기준 민감도", "",
         "기준을 바꾼 데이터로 학습·평가를 다시 했을 때 Baseline 1 대비 M2의 개선이 유지되는가:", "",
         to_md(pd.DataFrame(sens_rows)), "",
         "- 3시간 초과를 빼면 긴 꼬리가 줄어 두 모델의 MAE가 모두 작아진다. 비교할 것은 절대값이 아니라 B1 대비 변화다.", "",
         "## 3. 특성 중요도 (M2)", "", f"![중요도]({f_imp})", "", to_md(imp_fmt), "",
         "## 4. 오류 분석 (M2, 2025년 평가 데이터)", "", f"![구간별 오차]({f_err})", "",
         "실제 대여 시간 구간별 (평균 편향 = 예측 − 실제, 음수면 짧게 예측):", "", to_md(err_by("실제 구간"), index=True), "",
         "자전거 모델별:", "", to_md(err_by("bike_model"), index=True), "",
         "날짜 유형별:", "", to_md(err_by("day_type"), index=True), "",
         "왕복 여부별 (특성으로는 쓰지 않음, 오류 분석에만 사용):", "", to_md(err_by("is_round_trip"), index=True), "",
         "## 5. 관련성: 전기자전거는 같은 조건에서도 더 오래 타는가", "",
         "- 방법: 연도 × 시작 시간대 × 날짜 유형 × 출발 대여소 유형이 같은 '층' 안에서 전기자전거와 일반 자전거의 중앙값 차이를 구하고, "
         "두 모델 중 적은 쪽 건수로 가중 평균 (두 모델 모두 20건 이상인 층만)",
         "- 목적: 전기자전거가 특정 시간대·장소에 몰려서 생긴 차이인지(구성 효과), 같은 조건에서도 남는 차이인지 구분", "",
         to_md(pd.DataFrame(strat_rows)) if strat_rows else "- 비교할 수 있는 자전거 모델이 둘이 아님", "",
         "## 6. 해석 작성 시 확인할 것", "",
         "- 예측 성능이 Baseline보다 조금만 좋다면, '대여 시작 시점 정보로는 대여 시간을 많이 설명할 수 없다'는 것도 결과다.",
         "- 긴 대여(40분 이상)의 오차가 큰 것은 이동 목적·경로 정보가 없기 때문으로 보인다 (가설).",
         "- 층화 후에도 전기자전거 차이가 남는다면 시간·장소 구성만으로는 설명되지 않는다는 뜻이다. 거리 정보가 없어 원인(더 먼 이동 등)은 확인할 수 없다.",
         ""]
    (OUT_DIR / "q3_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[완료] {OUT_DIR / 'q3_report.md'}")


if __name__ == "__main__":
    main()