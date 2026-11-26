"""
1차 피드백 추가 점검

1차 피드백에서 요구했지만 아직 계산하지 않은 항목을 한 번에 점검한다.
  1. 대여시간의 평균과 중앙값 비교 (피드백: "평균과 중앙값을 함께 확인")
  2. 대여소별 요약표: 출발·도착 건수와 차이, 평일·주말 일평균, 동일 대여소 반납 비율, 대여시간 중앙값
     + OD 집계 합계 = 출발 합계 = 도착 합계 검산
  3. 2023년 8월만 본 결과 vs 3년 결과 (피드백: 범위를 2023년 8월로 명확히 → 3년으로 확장한 결정이 결론을 바꾸는지 확인)
  4. 피드백이 제안한 특징 세트(표준화)로 K-means를 하고, 시간대 프로파일 군집(질문 1)과 비교

입력: data/processed/trips/ (한 번 읽음), data/agg/ (03_eda.py, 04~05 결과)
출력: data/agg/station_summary.csv, 03_analysis-modeling/feedback_checks.md, figures/fb_*.png

사용법 (proj 폴더에서 실행, 05_q1_station_types.py 실행 후)
    python 08_feedback_checks.py
"""

import gc
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from common import (AGG_DIR, AM_PEAK, C, HOLIDAY_DATES, OUT_DIR, OUTAGE_DATES, PM_PEAK, TRIPS_DIR, day_counts,
                    finish, fmt_int, load_station_hourly, new_ax, setup_font, station_metrics, station_profiles, to_md)

DUR_BIN = 0.1                    # 대여소별 대여시간 히스토그램 간격(분)
DUR_CAP = 1800                   # 180분 이상은 한 칸으로 (중앙값 계산에는 영향 없음)
AUG_YEAR, AUG_MONTH = 2023, 8
MIN_AUG_TRIPS = 200              # 8월 한 달 출발+도착이 이 이상인 대여소만 유형 비교
MIN_BAND_PER_DAY = 3             # 질문 2와 같은 판단 기준
STRONG_INDEX = 0.5               # 질문 2와 같은 '크게 불균형' 기준 (3배)
K_RANGE = range(2, 7)
SEED = 0

FEATURES = {  # 피드백 표의 특징 → 계산 방법
    "log_trips_per_workday": "평일 일평균 이용량 (출발+도착, log)",
    "log_trips_per_nonworkday": "주말·공휴일 일평균 이용량 (출발+도착, log)",
    "am_arrival_ratio": "평일 오전 첨두 도착 비중",
    "pm_arrival_ratio": "평일 오후 첨두 도착 비중",
    "day_index": "출발·도착 차이 (도착−출발)/(도착+출발), 전체 기간",
    "round_trip_share": "동일 대여소 반납 비율",
    "log_median_duration": "대여시간 중앙값 (분, log)",
}


class Acc:
    def __init__(self):
        self.store = {}

    def add(self, name, obj):
        self.store[name] = obj if name not in self.store else self.store[name].add(obj, fill_value=0)


# ---------------------------------------------------------------------------
# 1) 원자료 한 번 읽기
# ---------------------------------------------------------------------------
def aggregate():
    files = sorted(TRIPS_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"분석용 데이터가 없습니다: {TRIPS_DIR}")
    cols = ["start_station_key", "end_station_key", "is_round_trip", "duration_min", "bike_model", "year", "month",
            "start_date", "start_hour", "day_of_week", "end_time"]
    acc, codes, rows, outage = Acc(), {}, 0, 0
    for i, path in enumerate(files, 1):
        df = pq.read_table(path, columns=cols).to_pandas()
        o = df["start_date"].isin(OUTAGE_DATES)
        outage += int(o.sum())
        df = df[~o]
        rows += len(df)
        for k in pd.unique(pd.concat([df["start_station_key"], df["end_station_key"]], ignore_index=True)):
            codes.setdefault(k, len(codes))
        df["s"] = df["start_station_key"].map(codes).astype("int32")
        df["e"] = df["end_station_key"].map(codes).astype("int32")
        df["rt"] = df["is_round_trip"].astype(bool)
        df["model"] = df["bike_model"].astype(str)
        df["workday"] = (df["day_of_week"] < 5) & ~df["start_date"].isin(HOLIDAY_DATES)

        acc.add("dur_sum", df.groupby(["year", "model", "workday", "rt"])["duration_min"].agg(["size", "sum"]))
        acc.add("st_rt", df.groupby("s")["rt"].agg(["size", "sum"]))
        acc.add("st_dur_sum", df.groupby("s")["duration_min"].sum())
        dbin = np.minimum(np.floor(df["duration_min"] / DUR_BIN), DUR_CAP).astype("int32")
        acc.add("st_hist", df.groupby([df["s"], dbin.rename("dbin")]).size())

        aug = df[(df["year"] == AUG_YEAR) & (df["month"] == AUG_MONTH)]
        if len(aug):
            acc.add("aug_start", aug.groupby(["s", "workday", "start_hour"]).size())
            acc.add("aug_end", aug.groupby(["e", "workday", aug["end_time"].dt.hour.rename("end_hour")]).size())
        del df, aug
        gc.collect()
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")
    names = pd.Series({v: k for k, v in codes.items()})
    return acc.store, names, rows, outage


def hist_median(h: pd.Series) -> float:
    """0.1분 구간 히스토그램(인덱스 = 구간 번호)의 중앙값 (구간 중앙값)."""
    h = h.sort_index()
    cum = h.cumsum() / h.sum()
    b = int(h.index[np.searchsorted(cum.to_numpy(), 0.5)])
    return (b + 0.5) * DUR_BIN


# ---------------------------------------------------------------------------
# 2) 각 점검
# ---------------------------------------------------------------------------
def check_mean_median(store):
    s = store["dur_sum"].reset_index()
    s.columns = ["year", "model", "workday", "rt", "n", "total"]
    eda = pd.read_parquet(AGG_DIR / "duration_hist.parquet")  # 03_eda.py (장애일 184건 포함)
    eda["model"] = eda["bike_model"].astype(str)

    groups = [("전체", None), ("평일", ("workday", True)), ("주말·공휴일", ("workday", False)),
              ("편도", ("rt", False)), ("왕복 (출발한 대여소에 반납)", ("rt", True))]
    rows = []
    for label, cond in groups:
        a = s if cond is None else s[s[cond[0]] == cond[1]]
        col = {"workday": "workday", "rt": "is_round_trip"}.get(cond[0]) if cond else None
        h = eda if cond is None else eda[eda[col] == cond[1]]
        rows.append((label, a["n"].sum(), a["total"].sum() / a["n"].sum(), hist_median(h.groupby("dbin")["trips"].sum())))
    for (y, m), a in s.groupby(["year", "model"]):
        h = eda[(eda["year"] == y) & (eda["model"] == m)]
        rows.append((f"{y}년 {m}", a["n"].sum(), a["total"].sum() / a["n"].sum(),
                     hist_median(h.groupby("dbin")["trips"].sum())))
    t = pd.DataFrame(rows, columns=["구분", "대여 건수", "평균(분)", "중앙값(분)"])
    t["평균 ÷ 중앙값"] = (t["평균(분)"] / t["중앙값(분)"]).round(2)
    t["대여 건수"] = t["대여 건수"].map(fmt_int)
    t["평균(분)"] = t["평균(분)"].round(2)
    t["중앙값(분)"] = t["중앙값(분)"].round(1)
    return t


def build_station_summary(store, names):
    sh = load_station_hourly("base")
    years = sorted(sh["year"].unique())
    tot = sh.groupby("station_key")[["starts", "ends"]].sum()
    per_year = sh.groupby(["station_key", "year"])[["starts", "ends"]].sum().sum(axis=1).unstack(fill_value=0)
    m = station_metrics(sh)

    rt = store["st_rt"].rename(index=names)
    rt.columns = ["n_start", "n_rt"]
    dur_mean = (store["st_dur_sum"] / store["st_rt"]["size"]).rename(index=names)
    hist = store["st_hist"]
    med = hist.groupby(level=0).apply(lambda h: hist_median(h.droplevel(0))).rename(index=names)

    out = tot.join(m, how="left")
    out["diff_ends_minus_starts"] = out["ends"] - out["starts"]
    out["day_index"] = out["diff_ends_minus_starts"] / (out["ends"] + out["starts"]).where(lambda x: x > 0)
    out["round_trip_share"] = (rt["n_rt"] / rt["n_start"]).reindex(out.index)
    out["median_duration_min"] = med.reindex(out.index)
    out["mean_duration_min"] = dur_mean.reindex(out.index)
    out["eligible"] = (per_year.reindex(out.index).reindex(columns=years, fill_value=0) >= 1000).all(axis=1)
    cpath = AGG_DIR / "station_clusters.csv"
    if cpath.exists():
        cl = pd.read_csv(cpath).set_index("station_key")
        name_col = "cluster_name" if "cluster_name" in cl.columns else "suggested_label"
        out = out.join(cl[["cluster", name_col]].rename(columns={name_col: "cluster_name"}), how="left")
    out.index.name = "station_key"
    return out.reset_index()


def check_od_totals(summary):
    od = pd.read_parquet(AGG_DIR / "od.parquet")
    st = pd.read_parquet(AGG_DIR / "station_hourly.parquet")
    vals = {"OD 집계 합계": int(od["trips"].sum()), "대여소 출발 합계": int(st["starts"].sum()),
            "대여소 도착 합계": int(st["ends"].sum())}
    return vals, len(set(vals.values())) == 1


def check_august(store, names):
    # (a) 시간대 프로파일 (03_eda.py 일별·시간대 집계)
    hourly = pd.read_parquet(AGG_DIR / "hourly.parquet")
    hourly = hourly[~hourly["date"].isin(OUTAGE_DATES)]
    hourly["aug"] = (hourly["date"].dt.year == AUG_YEAR) & (hourly["date"].dt.month == AUG_MONTH)
    prof_rows, profiles = [], {}
    for wd, label in [(True, "평일"), (False, "주말·공휴일")]:
        for scope, mask in [("3년", np.ones(len(hourly), bool)), (f"{AUG_YEAR}년 {AUG_MONTH}월", hourly["aug"])]:
            h = hourly[mask & (hourly["workday"] == wd)]
            p = h.groupby("hour")["trips"].sum() / h["date"].nunique()
            profiles[(label, scope)] = p
            top = p.nlargest(3).index.tolist()
            prof_rows.append({"날짜 유형": label, "범위": scope, "하루 평균": fmt_int(p.sum()),
                              "상위 3개 시간대": ", ".join(f"{x}시대" for x in top),
                              "7~9시대 비중": f"{p[AM_PEAK].sum() / p.sum() * 100:.1f}%",
                              "16~18시대 비중": f"{p[PM_PEAK].sum() / p.sum() * 100:.1f}%"})
    corr = {lbl: profiles[(lbl, "3년")].corr(profiles[(lbl, f"{AUG_YEAR}년 {AUG_MONTH}월")])
            for lbl in ["평일", "주말·공휴일"]}

    # (b) 대여소 오전 첨두 불균형
    a_s = store["aug_start"].rename("starts")
    a_s.index.names = ["code", "workday", "hour"]
    a_e = store["aug_end"].rename("ends")
    a_e.index.names = ["code", "workday", "hour"]
    aug = pd.concat([a_s, a_e], axis=1).fillna(0).reset_index()
    aug.insert(0, "station_key", aug["code"].map(names))
    aug = aug.drop(columns="code")
    aug["year"] = AUG_YEAR
    cal = pd.DataFrame({"date": pd.date_range(f"{AUG_YEAR}-{AUG_MONTH:02d}-01", periods=31)})
    aug_workdays = int(((cal["date"].dt.dayofweek < 5) & ~cal["date"].isin(HOLIDAY_DATES)).sum())

    def am_table(df, n_days):
        g = df[df["workday"] & df["hour"].isin(AM_PEAK)].groupby("station_key")[["starts", "ends"]].sum()
        g["vol"] = (g["starts"] + g["ends"]) / n_days
        g["net"] = (g["ends"] - g["starts"]) / n_days
        g["index"] = (g["ends"] - g["starts"]) / (g["starts"] + g["ends"]).where(lambda x: x > 0)
        return g[g["vol"] >= MIN_BAND_PER_DAY]

    sh = load_station_hourly("base")
    n_wd = int(day_counts().query("workday")["days"].sum())
    full, aug_am = am_table(sh, n_wd), am_table(aug, aug_workdays)
    j = full[["net"]].join(aug_am[["net"]], lsuffix="_3y", rsuffix="_aug", how="inner")
    rho = j["net_3y"].rank().corr(j["net_aug"].rank())
    top_out = len(set(full["net"].nsmallest(10).index) & set(aug_am["net"].nsmallest(10).index))
    top_in = len(set(full["net"].nlargest(10).index) & set(aug_am["net"].nlargest(10).index))
    imb_rows = [{"범위": "3년", "판단 대상": len(full),
                 "크게 불균형 (|I| ≥ 0.5)": f"{(full['index'].abs() >= STRONG_INDEX).sum()} ({(full['index'].abs() >= STRONG_INDEX).mean() * 100:.0f}%)",
                 "출발 우세 1위": full["net"].idxmin(), "도착 우세 1위": full["net"].idxmax()},
                {"범위": f"{AUG_YEAR}년 {AUG_MONTH}월 (평일 {aug_workdays}일)", "판단 대상": len(aug_am),
                 "크게 불균형 (|I| ≥ 0.5)": f"{(aug_am['index'].abs() >= STRONG_INDEX).sum()} ({(aug_am['index'].abs() >= STRONG_INDEX).mean() * 100:.0f}%)",
                 "출발 우세 1위": aug_am["net"].idxmin(), "도착 우세 1위": aug_am["net"].idxmax()}]

    # (c) 질문 1 유형: 8월 프로파일을 3년 군집 중심에 배정
    cl_res = None
    cpath, cent_path = AGG_DIR / "station_clusters.csv", AGG_DIR / "station_cluster_centroids.csv"
    if cpath.exists() and cent_path.exists():
        cl = pd.read_csv(cpath).set_index("station_key")
        cent = pd.read_csv(cent_path).set_index("cluster")
        vol = aug.groupby("station_key")[["starts", "ends"]].sum().sum(axis=1)
        st = [s for s in cl.index if vol.get(s, 0) >= MIN_AUG_TRIPS]
        Xa = station_profiles(aug, stations=st).to_numpy()
        d = ((Xa[:, None, :] - cent.to_numpy()[None, :, :]) ** 2).sum(axis=2)
        pred = cent.index.to_numpy()[d.argmin(axis=1)]
        true = cl.loc[st, "cluster"].to_numpy()
        cl_res = {"n": len(st), "agree": (pred == true).mean(), "ari": adjusted_rand_score(true, pred)}

    return pd.DataFrame(prof_rows), corr, pd.DataFrame(imb_rows), rho, len(j), top_out, top_in, cl_res, profiles


def check_feature_clustering(summary):
    cl_path = AGG_DIR / "station_clusters.csv"
    if not cl_path.exists():
        return None
    s = summary[summary["eligible"] & summary["cluster"].notna()].set_index("station_key").copy()
    s["log_trips_per_workday"] = np.log1p(s["trips_per_workday"])
    s["log_trips_per_nonworkday"] = np.log1p(s["trips_per_nonworkday"])
    s["log_median_duration"] = np.log(s["median_duration_min"])
    F = s[list(FEATURES)].copy()
    F = F.fillna(F.median())
    Z = StandardScaler().fit_transform(F.to_numpy())

    ksel = []
    fits = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(Z)
        order = pd.Series(km.labels_).value_counts().index.tolist()
        lab = np.array([{o: i for i, o in enumerate(order)}[l] for l in km.labels_])
        fits[k] = (km, order, lab)
        ksel.append({"k": k, "실루엣": round(silhouette_score(Z, km.labels_), 3),
                     "질문 1 유형(k=3)과의 ARI": round(adjusted_rand_score(s["cluster"], lab), 3)})
    ksel = pd.DataFrame(ksel)
    k_sil = int(ksel.loc[ksel["실루엣"].idxmax(), "k"])

    def describe(k):
        km, order, lab = fits[k]
        d = s.assign(fc=lab)
        g = d.groupby("fc").agg(대여소=("trips_per_day", "size"),
                                평일_일평균=("trips_per_workday", "median"), 주말공휴일_일평균=("trips_per_nonworkday", "median"),
                                비근무일_근무일_비=("nonwork_rate", "median"), 오전도착비중=("am_arrival_ratio", "median"),
                                오후도착비중=("pm_arrival_ratio", "median"), 반납비율=("round_trip_share", "median"),
                                대여시간_중앙값=("median_duration_min", "median"))
        g = g.round({"평일_일평균": 1, "주말공휴일_일평균": 1, "비근무일_근무일_비": 2, "오전도착비중": 2, "오후도착비중": 2,
                     "반납비율": 3, "대여시간_중앙값": 1})
        top = d.sort_values("trips_per_day", ascending=False).groupby("fc").head(4)
        g["이용 상위 대여소"] = top.groupby("fc").apply(lambda x: " / ".join(i.split(",")[0] for i in x.index))
        cross = pd.crosstab(d["fc"].rename("특징 군집"), d["cluster_name"].rename("질문 1 유형"))
        centers = pd.DataFrame(km.cluster_centers_[order], columns=[FEATURES[f] for f in FEATURES])
        return g, cross, centers

    results = {k: describe(k) for k in sorted({3, k_sil})}
    return ksel, k_sil, results, len(s)


def plot_centers(centers: pd.DataFrame, k: int, filename: str):
    cmap = LinearSegmentedColormap.from_list("div", [C["neg"], "#f1f0ec", C["s1"]])
    fig, ax = new_ax(10, 1.2 + 0.55 * k)
    ax.grid(False)
    v = np.abs(centers.to_numpy()).max()
    im = ax.imshow(centers.to_numpy(), cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    ax.set_yticks(range(k), [f"특징 군집 {i}" for i in range(k)])
    ax.set_xticks(range(centers.shape[1]), [c.split(" (")[0] for c in centers.columns], rotation=25, ha="right", fontsize=8)
    for i in range(k):
        for j in range(centers.shape[1]):
            ax.text(j, i, f"{(0.0 if abs(centers.iat[i, j]) < 0.05 else centers.iat[i, j]):+.1f}", ha="center", va="center", fontsize=8, color=C["ink"])
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("표준화 값 (0 = 전체 평균)", color=C["ink2"], fontsize=8)
    cb.outline.set_visible(False)
    return finish(fig, f"피드백 특징 세트로 한 군집의 중심 (k = {k})",
                  "파랑: 평균보다 높음 / 빨강: 평균보다 낮음 (표준화 값, 단위 = 표준편차)", filename)


# ---------------------------------------------------------------------------
def main():
    setup_font()
    print("[1/3] 원자료 집계")
    store, names, rows, outage = aggregate()
    print("[2/3] 점검")
    mm = check_mean_median(store)
    summary = build_station_summary(store, names)
    summary.to_csv(AGG_DIR / "station_summary.csv", index=False, encoding="utf-8-sig")
    od_vals, od_ok = check_od_totals(summary)
    prof_tab, prof_corr, imb_tab, rho, n_rho, top_out, top_in, cl_res, profiles = check_august(store, names)
    fc = check_feature_clustering(summary)

    print("[3/3] 그래프·보고서")
    # 그래프: 8월 vs 3년 시간대 프로파일 (같은 날짜 유형끼리, 하루 합 = 100%로 정규화해 모양 비교)
    fig, axes = new_ax(10, 3.6, ncols=2, sharey=True)
    for ax, lbl in zip(axes, ["평일", "주말·공휴일"]):
        for scope, color in [("3년", C["s1"]), (f"{AUG_YEAR}년 {AUG_MONTH}월", C["s2"])]:
            p = profiles[(lbl, scope)]
            ax.plot(p.index, p / p.sum() * 100, color=color, linewidth=2, label=scope)
        ax.set_title(f"{lbl} (상관 {prof_corr[lbl]:.3f})", loc="left", fontsize=9, color=C["ink2"])
        ax.set_xticks(range(0, 24, 4), [f"{h}시" for h in range(0, 24, 4)])
    axes[0].set_ylabel("하루 대여 중 비율(%)", color=C["ink2"], fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=C["ink2"])
    f_prof = finish(fig, f"{AUG_YEAR}년 {AUG_MONTH}월과 3년 전체의 시간대 모양 비교", "하루 대여를 100%로 맞춰 모양만 비교",
                    "fb_august_profile.png")

    hist_total = int(pd.read_parquet(AGG_DIR / "duration_hist.parquet", columns=["trips"])["trips"].sum())
    L = ["# 1차 피드백 추가 점검", "",
         f"- 실행 일시: {datetime.now():%Y-%m-%d %H:%M}",
         f"- 분석 행: {rows:,} (운영 장애일 {outage:,}건 제외)",
         "- 1차 피드백에서 요구했지만 2·3차에서 계산하지 않은 항목을 점검한다. 이 보고서는 **관찰(수치)**을 기록한다.", "",
         "## 1. 대여시간의 평균과 중앙값", "",
         "평균은 원자료에서 정확히 계산했고, 중앙값은 0.1분 간격 히스토그램(`03_eda.py`)의 구간 중앙값이다 "
         f"(히스토그램에는 장애일 {hist_total - rows:,}건이 포함되어 건수가 조금 다르지만 중앙값에는 사실상 영향이 없다).", "",
         to_md(mm), "",
         "- 평균 ÷ 중앙값이 1보다 크면 긴 대여(오른쪽 꼬리)가 평균을 끌어올린 것이다.", "",
         "## 2. 대여소별 요약표와 OD 검산", "",
         f"- 전체 대여소 {len(summary):,}개의 요약표: `data/agg/station_summary.csv`",
         "- 열: 출발·도착 건수, 차이(도착 − 출발), 평일·주말·공휴일 일평균, 비근무일/근무일 비, 오전·오후 첨두 도착 비중, "
         "동일 대여소 반납 비율, 대여시간 중앙값·평균, 분석 대상 여부(연 1,000건 이상), 질문 1 유형", ""]
    e = summary[summary["eligible"]]
    q = e[["round_trip_share", "median_duration_min", "trips_per_workday", "trips_per_nonworkday"]].quantile(
        [0.1, 0.5, 0.9]).T
    q.index = ["동일 대여소 반납 비율", "대여시간 중앙값(분)", "평일 일평균 이용", "주말·공휴일 일평균 이용"]
    q.columns = ["10%", "중앙값", "90%"]
    L += [f"분석 대상 대여소 {len(e):,}개의 분포:", "", to_md(q.round(3), index=True), ""]
    for col, title in [("round_trip_share", "동일 대여소 반납 비율"), ("median_duration_min", "대여시간 중앙값")]:
        top = e.nlargest(8, col)[["station_key", col, "nonwork_rate", "cluster_name"]] if "cluster_name" in e else \
            e.nlargest(8, col)[["station_key", col, "nonwork_rate"]]
        L += [f"{title} 상위 8개 대여소:", "", to_md(top.round(3).rename(columns={
            "station_key": "대여소", col: title, "nonwork_rate": "비근무일/근무일 비", "cluster_name": "질문 1 유형"})), ""]
    L += ["OD 검산 (03_eda.py 집계):", "",
          to_md(pd.DataFrame({"항목": od_vals.keys(), "건수": [fmt_int(v) for v in od_vals.values()]})), "",
          f"- 결과: **{'일치' if od_ok else '불일치 — 확인 필요'}** (OD 행렬의 모든 칸을 더한 값 = 출발 합계 = 도착 합계)", "",
          f"## 3. {AUG_YEAR}년 {AUG_MONTH}월만 본 결과 vs 3년 결과", "",
          "1차 피드백은 한 달 자료로는 연중 특성을 설명하기 어려우니 범위를 2023년 8월로 명확히 설정하고 다른 월은 확장 과제로 검토하라고 했다. 3년으로 확장한 결정이 주요 결론을 바꾸는지 확인한다.", "",
          "### 3.1 시간대 패턴", "", f"![8월 vs 3년]({f_prof})", "", to_md(prof_tab), "",
          f"- 시간대별 하루 평균 이용(24개 값)의 상관: 평일 {prof_corr['평일']:.3f}, 주말·공휴일 {prof_corr['주말·공휴일']:.3f}", "",
          "### 3.2 평일 오전 첨두 불균형", "", to_md(imb_tab), "",
          f"- 두 범위 모두 판단 대상인 대여소 {n_rho}개의 순유입 순위 상관(스피어만): **{rho:.3f}**",
          f"- 출발 우세 상위 10개 겹침 {top_out}/10, 도착 우세 상위 10개 겹침 {top_in}/10", ""]
    if cl_res:
        L += ["### 3.3 질문 1 유형", "",
              f"- 8월 한 달 출발+도착 {MIN_AUG_TRIPS}건 이상인 대여소 {cl_res['n']}개의 8월 프로파일을 3년 군집 중심에 배정",
              f"- 3년 유형과 같은 유형으로 배정된 비율: **{cl_res['agree'] * 100:.1f}%** (ARI {cl_res['ari']:.3f})", ""]
    L += ["- 8월은 여름휴가철이라 평일 출퇴근 시간대 집중이 다른 달보다 약할 수 있다. 차이가 있다면 이 점을 함께 고려한다.", ""]

    if fc:
        ksel, k_sil, res, n_st = fc
        L += ["## 4. 피드백 특징 세트로 한 군집 vs 시간대 프로파일 군집", "",
              f"- 대상: 질문 1과 같은 대여소 {n_st}개",
              "- 특징 (피드백 표를 따름, **표준화: 평균 0·표준편차 1**):", ""]
        L += [f"  - {v}" for v in FEATURES.values()]
        L += ["", "- 이용량과 대여시간은 대여소마다 수십 배 차이가 나고 오른쪽 꼬리가 길어 log를 취한 뒤 표준화했다.",
              "- 단위가 다른 특징(건수, 비율, 분)을 함께 쓰므로 표준화가 필요하다. (질문 1의 시간대 프로파일은 모두 같은 단위의 비율이라 표준화하지 않았다.)", "",
              to_md(ksel), "", f"- 실루엣이 가장 높은 k: **{k_sil}**", ""]
        for k, (g, cross, centers) in res.items():
            fig_path = plot_centers(centers, k, f"fb_feature_centers_k{k}.png")
            leisure = g[g["비근무일_근무일_비"] >= 1.0]
            L += [f"### k = {k}", "", f"![k={k}]({fig_path})", "", to_md(g.rename_axis("특징 군집"), index=True), "",
                  f"- 비근무일 이용이 근무일 이상인 군집(중앙값 ≥ 1): "
                  + ("**있음** — " + ", ".join(f"군집 {i} ({int(r['대여소'])}개)" for i, r in leisure.iterrows())
                     if len(leisure) else "없음"), "",
                  "특징 군집 × 질문 1 유형 (대여소 수):", "", to_md(cross, index=True), ""]
    L += ["## 5. 해석 작성 시 확인할 것", "",
          "- 3절의 상관·일치율이 높으면 '3년으로 확장해도 2023년 8월과 같은 결론'이라고 쓸 수 있다.",
          "- 4절에서 비근무일 이용이 많은 군집이 생기면, 3차의 '여가 성격은 시간대 모양으로 갈라지지 않는다'는 결과가 "
          "**특징 설계 때문**이었음을 뜻한다. 반납 비율·대여시간이 그 군집을 가르는 특징인지 중심 그래프로 확인한다.",
          "- 반납 비율이 높은 대여소의 목적(여가 등)은 데이터로 확인할 수 없다. '출발한 대여소에 반납한 비율이 높다'까지가 관찰이다.", ""]
    (OUT_DIR / "feedback_checks.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[완료] {OUT_DIR / 'feedback_checks.md'}")


if __name__ == "__main__":
    main()