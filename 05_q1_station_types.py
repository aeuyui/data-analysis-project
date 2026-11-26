"""
3차 2단계 — 질문 1: 대여소를 이용 패턴에 따라 몇 개 유형으로 나눌 수 있는가?

방법
- 대여소마다 96차원 프로파일을 만든다: (출발/도착) × (근무일/비근무일) × 24시간의 비율.
  규모가 아니라 '언제 출발·도착이 몰리는가'의 모양을 비교하기 위해 대여소 전체 대여로 나눈다.
- 3년 모두 충분히 이용된 대여소만 쓴다 (연 1,000건 이상): 이용이 적으면 비율이 우연에 크게 흔들린다.
- K-means로 군집화한다. k = 2~8의 엘보·실루엣을 기록하고, k = 2~5는 군집 내용(요약표, 규칙 교차표,
  프로파일 그래프)을 나란히 비교한다. 실루엣은 '가장 뚜렷한 구분'을, 내용 비교는 '해석 가능한 세부 유형'을 보여 준다.
- 최종 k: K_FIXED가 None이면 실루엣 최대, 숫자면 그 값 (비교 결과를 보고 근거와 함께 지정)

검증 (최종 k에 대해)
1. 시드 안정성: 초기값(random_state)을 바꿔도 같은 군집이 나오는가 (ARI)
2. 연도 안정성: 각 연도의 프로파일을 같은 중심점에 배정했을 때 유형이 유지되는가 (일치율, ARI)
3. 전처리 기준 민감도: 짧은 왕복 2분·긴 대여 3시간 기준으로 바꾼 데이터에서도 유형이 유지되는가
4. 단순 규칙과 비교: '비근무일 이용 비율, 오전 도착 비율'만으로 나눈 규칙과 얼마나 같은가

입력: data/agg/station_hourly_year.parquet (04_build_analysis_tables.py)
출력: data/agg/station_clusters.csv, data/agg/station_cluster_centroids.csv,
      03_analysis-modeling/q1_report.md, 03_analysis-modeling/figures/q1_*.png

사용법 (proj 폴더에서 실행)
    pip install scikit-learn
    python 05_q1_station_types.py
"""

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from common import (AGG_DIR, C, OUT_DIR, finish, fmt_int, load_station_hourly, new_ax, setup_font,
                    station_metrics, station_profiles, to_md)

MIN_TRIPS_PER_YEAR = 1000      # 연도마다 출발+도착이 이 이상인 대여소만 군집화
K_RANGE = range(2, 9)          # 엘보·실루엣 기록 범위
K_COMPARE = [2, 3, 4, 5]       # 군집 내용을 나란히 비교할 k
# 최종 k = 3 (2026-11-23 k 비교 결과로 결정, 3차 문서 4.5절)
#  - 실루엣은 k = 2(0.435)가 최대지만, k = 2의 큰 군집에 혼합형·비근무일형이 섞여 있음
#  - k = 3이 단순 규칙과의 일치(ARI 0.559)가 가장 높고, 세 군집의 시간대 모양이 뚜렷이 다름
#  - k = 4·5는 같은 방향의 군집을 첨두 높이에 따라 더 쪼갤 뿐 새로운 모양이 나오지 않음
K_FIXED = 3                    # None이면 실루엣 최대 k
# 그래프와 지표를 보고 정한 군집 이름 (k = 3일 때만 적용, 군집 번호는 대여소 수가 많은 순서)
# 1차 피드백에 따라 장소 추정("도심" 등)이 아니라 관찰된 특성으로 이름을 붙인다.
CLUSTER_NAMES = {3: {0: "평일 오전 출발 집중형", 1: "시간대 분산형", 2: "평일 오전 도착 집중형"}}
# 이름이 군집 내용과 맞는지 확인하는 조건 (오전 첨두 도착 비율 중앙값). 맞지 않으면 경고한다.
NAME_CHECKS = {"평일 오전 출발 집중형": lambda am: am <= 0.4, "시간대 분산형": lambda am: 0.4 < am < 0.7,
               "평일 오전 도착 집중형": lambda am: am >= 0.7}
SEEDS = range(1, 11)

# 단순 규칙 (군집화 결과와 비교용)
RULE_NONWORK_RATE = 1.0        # 비근무일 하루 평균 이용 ≥ 근무일 하루 평균 → 비근무일형
RULE_AM_HIGH, RULE_AM_LOW = 0.6, 0.4


def fit_kmeans(X, k, seed=0, n_init=30):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)


def fit_ordered(X, k, n_init=50):
    """군집 번호를 대여소 수가 많은 순서로 0, 1, 2 … 로 다시 매긴다."""
    km = fit_kmeans(X, k, n_init=n_init)
    order = pd.Series(km.labels_).value_counts().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in km.labels_])

    def predict(Xn):
        return np.array([remap[l] for l in km.predict(Xn)])

    return km, order, labels, predict


def rule_type(m: pd.DataFrame) -> pd.Series:
    return pd.Series(np.select(
        [m["nonwork_rate"] >= RULE_NONWORK_RATE, m["am_arrival_ratio"] >= RULE_AM_HIGH,
         m["am_arrival_ratio"] <= RULE_AM_LOW],
        ["비근무일형", "오전 도착형", "오전 출발형"], default="혼합형"), index=m.index)


def suggest_label(row) -> str:
    """군집 중앙값으로 붙이는 '제안' 이름. 보고서에서 그래프를 보고 확정한다."""
    if row["nonwork_rate"] >= 1.0:
        return "여가형 (비근무일 이용이 많음)"
    if row["am_arrival_ratio"] >= 0.55:
        return "도착지형 (평일 오전 도착 우세)"
    if row["am_arrival_ratio"] <= 0.45:
        return "출발지형 (평일 오전 출발 우세)"
    return "혼합형"


def summarize(m: pd.DataFrame, labels) -> pd.DataFrame:
    mm = m.assign(cluster=labels)
    s = mm.groupby("cluster").agg(
        대여소=("trips_per_day", "size"), 하루평균_중앙값=("trips_per_day", "median"),
        nonwork_rate=("nonwork_rate", "median"), am_arrival_ratio=("am_arrival_ratio", "median"),
        pm_arrival_ratio=("pm_arrival_ratio", "median"))
    s["이용 비중"] = (mm.groupby("cluster")["trips_per_day"].sum() / mm["trips_per_day"].sum() * 100).round(1).astype(str) + "%"
    s["제안 이름"] = s.apply(suggest_label, axis=1)
    top = mm.sort_values("trips_per_day", ascending=False).groupby("cluster").head(5)
    s["이용 상위 대여소"] = top.groupby("cluster").apply(lambda g: " / ".join(g.index))
    return s


def fmt_summary(s: pd.DataFrame) -> pd.DataFrame:
    f = s.copy()
    f["하루평균_중앙값"] = f["하루평균_중앙값"].round(1)
    for col in ["nonwork_rate", "am_arrival_ratio", "pm_arrival_ratio"]:
        f[col] = f[col].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    return f.rename(columns={"하루평균_중앙값": "하루 평균 대여(중앙값)", "nonwork_rate": "비근무일/근무일 이용 비",
                             "am_arrival_ratio": "오전 첨두 도착 비율", "pm_arrival_ratio": "오후 첨두 도착 비율"})


def plot_profiles(X_df, labels, summary, k, filename, title):
    ncols = min(k, 5)
    nrows = int(np.ceil(k / ncols))
    fig, axes = new_ax(3.0 * ncols, 2.9 * nrows + 0.6, ncols=ncols, nrows=nrows, sharey=True, squeeze=False)
    hours = np.arange(24)
    for c in range(k):
        ax = axes.flat[c]
        # 열 순서: (출발, 도착) × (근무일, 비근무일) × 24시간 → [종류, 근무일 여부, 시간] 배열로 바꿈
        prof = X_df[labels == c].mean().to_numpy().reshape(2, 2, 24) * 100
        ax.plot(hours, prof[0, 0], color=C["s1"], linewidth=2, label="근무일 출발")
        ax.plot(hours, prof[1, 0], color=C["s2"], linewidth=2, label="근무일 도착")
        ax.plot(hours, prof[0, 1] + prof[1, 1], color=C["s3"], linewidth=2, label="비근무일 출발+도착")
        ax.set_title(f"군집 {c} · {(labels == c).sum()}개\n{summary.loc[c, '제안 이름']}", loc="left",
                     fontsize=9, color=C["ink"])
        ax.set_xticks(range(0, 24, 6), [f"{h}시" for h in range(0, 24, 6)])
    for ax in list(axes.flat)[k:]:
        ax.set_visible(False)
    axes.flat[0].set_ylabel("대여소 전체 대여 중 비율(%)", color=C["ink2"], fontsize=9)
    handles, lbls = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=C["ink2"])
    return finish(fig, title, "각 대여소의 전체 대여(3년 합)를 100%로 둔 시간대별 비율의 군집 평균", filename, bottom_in=0.35)


def main():
    setup_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sh = load_station_hourly("base")
    years = sorted(sh["year"].unique())

    # 대상 대여소
    tot = sh.groupby(["station_key", "year"])[["starts", "ends"]].sum().sum(axis=1).unstack(fill_value=0)
    tot = tot.reindex(columns=years, fill_value=0)
    eligible = tot.index[(tot >= MIN_TRIPS_PER_YEAR).all(axis=1)]
    X_df = station_profiles(sh, stations=eligible)
    X = X_df.to_numpy()
    m = station_metrics(sh).loc[eligible]
    m["rule"] = rule_type(m)
    print(f"대상 대여소 {len(eligible)} / 전체 {len(tot)}")

    # k별 엘보·실루엣
    rows = []
    for kk in K_RANGE:
        km_ = fit_kmeans(X, kk)
        rows.append({"k": kk, "inertia": km_.inertia_, "silhouette": silhouette_score(X, km_.labels_)})
    ksel = pd.DataFrame(rows)
    k_sil = int(ksel.loc[ksel["silhouette"].idxmax(), "k"])
    k = K_FIXED or k_sil

    # k별 군집 내용 비교
    compare = {}
    for kk in K_COMPARE:
        _, _, lab, _ = fit_ordered(X, kk)
        s = summarize(m, lab)
        compare[kk] = {
            "summary": s, "labels": lab,
            "rule_ari": adjusted_rand_score(m["rule"], lab),
            "cross": pd.crosstab(pd.Series(lab, index=m.index, name="cluster"), m["rule"]),
            "fig": plot_profiles(X_df, lab, s, kk, f"q1_profiles_k{kk}.png", f"군집별 평균 시간대 프로파일 (k = {kk})"),
        }

    # 최종 모델
    km, order, labels, predict = fit_ordered(X, k)
    summary = summarize(m, labels)
    names = CLUSTER_NAMES.get(k)
    name_warnings = []
    if names:
        summary["제안 이름"] = summary.index.map(names)   # 직접 정한 이름으로 바꿈
        for c, n in names.items():
            am = summary.loc[c, "am_arrival_ratio"]
            if n in NAME_CHECKS and not (pd.notna(am) and NAME_CHECKS[n](am)):
                name_warnings.append(f"군집 {c}의 이름 '{n}'이 오전 첨두 도착 비율 중앙값({am:.2f})과 맞지 않음")
        for w in name_warnings:
            print(f"[경고] {w} → CLUSTER_NAMES를 다시 확인하세요")
    m["cluster"] = labels
    m["suggested_label"] = m["cluster"].map(summary["제안 이름"])
    m["cluster_name"] = m["suggested_label"]

    # 1. 시드 안정성
    seed_ari = [adjusted_rand_score(labels, fit_kmeans(X, k, seed=s).labels_) for s in SEEDS]

    # 2. 연도 안정성
    year_rows = []
    for y in years:
        Xy = station_profiles(sh, years=[y], stations=eligible).to_numpy()
        py = predict(Xy)
        refit = fit_kmeans(Xy, k).labels_
        year_rows.append({"연도": y, "같은 중심점에 배정했을 때 유형 일치율": f"{(py == labels).mean() * 100:.1f}%",
                          "ARI (배정)": round(adjusted_rand_score(labels, py), 3),
                          "ARI (그 해만으로 다시 군집화)": round(adjusted_rand_score(labels, refit), 3)})

    # 3. 전처리 기준 민감도
    var_rows = []
    for v, desc in [("short2", "짧은 왕복 2분 이하 추가 제외"), ("long3h", "3시간 초과 추가 제외")]:
        Xv = station_profiles(load_station_hourly(v), stations=eligible).to_numpy()
        pv = predict(Xv)
        var_rows.append({"변형": desc, "유형 일치율": f"{(pv == labels).mean() * 100:.1f}%",
                         "ARI (배정)": round(adjusted_rand_score(labels, pv), 3),
                         "ARI (다시 군집화)": round(adjusted_rand_score(labels, fit_kmeans(Xv, k).labels_), 3)})

    # 4. 단순 규칙과 비교
    rule_ari = adjusted_rand_score(m["cluster"], m["rule"])
    cross = pd.crosstab(m["cluster"], m["rule"])

    # 저장
    out = m.reset_index().rename(columns={"index": "station_key"})
    out.to_csv(AGG_DIR / "station_clusters.csv", index=False, encoding="utf-8-sig")
    cent = pd.DataFrame(km.cluster_centers_[order], columns=[f"{a}_{'work' if b else 'nonwork'}_{c:02d}"
                                                             for a, b, c in X_df.columns])
    cent.insert(0, "cluster", range(k))
    cent.to_csv(AGG_DIR / "station_cluster_centroids.csv", index=False, encoding="utf-8-sig")

    # 그래프: k 선택 (두 지표를 한 축에 겹치지 않고 나란히)
    fig, axes = new_ax(10, 3.6, ncols=2)
    axes[0].plot(ksel["k"], ksel["inertia"], color=C["s1"], linewidth=2, marker="o", markersize=5)
    axes[0].set_title("엘보: 군집 내 제곱합", loc="left", fontsize=10, color=C["ink2"])
    axes[1].plot(ksel["k"], ksel["silhouette"], color=C["s1"], linewidth=2, marker="o", markersize=5)
    axes[1].set_title("실루엣 점수 (높을수록 잘 나뉨)", loc="left", fontsize=10, color=C["ink2"])
    for ax in axes:
        ax.axvline(k, color=C["muted"], linewidth=1, linestyle="--")
        ax.set_xlabel("군집 수 k", color=C["ink2"], fontsize=9)
    f_k = finish(fig, "군집 수 선택", f"점선: 최종 k = {k} ({'실루엣 최대' if K_FIXED is None else '직접 지정'}, 실루엣 최대는 k = {k_sil})",
                 "q1_k_selection.png")
    f_p = plot_profiles(X_df, labels, summary, k, "q1_cluster_profiles.png", f"군집별 평균 시간대 프로파일 (최종 k = {k})")

    # 보고서
    comp_tab = pd.DataFrame([{
        "k": kk, "실루엣": round(float(ksel.loc[ksel["k"] == kk, "silhouette"].iloc[0]), 3),
        "규칙과의 ARI": round(v["rule_ari"], 3),
        "군집 크기": " / ".join(str(n) for n in v["summary"]["대여소"]),
        "제안 이름": " / ".join(v["summary"]["제안 이름"].str.split(" ").str[0]),
    } for kk, v in compare.items()])

    L = ["# 질문 1. 대여소 이용 패턴 유형", "",
         f"- 실행 일시: {datetime.now():%Y-%m-%d %H:%M}",
         "- 질문: 시간대와 요일에 따라 이용량은 어떻게 다르며, 대여소를 이용 패턴에 따라 몇 개 유형으로 나눌 수 있는가?",
         "- 이 보고서는 **관찰(수치)**을 기록한다. 군집 이름은 자동 '제안'이며 그래프를 보고 확정한다.", "",
         "## 1. 대상과 특성", "",
         f"- 대상: 2023~2025년 매년 출발+도착 {fmt_int(MIN_TRIPS_PER_YEAR)}건 이상인 대여소 **{len(eligible)}개** "
         f"(전체 {len(tot)}개 중 {len(tot) - len(eligible)}개 제외)",
         f"- 대상 대여소가 차지하는 대여 비중: {tot.loc[eligible].to_numpy().sum() / tot.to_numpy().sum() * 100:.1f}% (출발+도착 기준)",
         "- 특성: (출발/도착) × (근무일/비근무일) × 24시간 = 96개 비율. 대여소 규모의 영향을 없애고 모양만 비교한다.",
         "- 운영 장애일(2025-08-05) 제외, 근무일 = 월~금 중 공휴일이 아닌 날", "",
         "## 2. 군집 수 선택", "", f"![k 선택]({f_k})", "",
         to_md(ksel.assign(inertia=ksel["inertia"].map(lambda v: f"{v:.4g}"), silhouette=ksel["silhouette"].round(3))), "",
         f"- 실루엣 점수가 가장 높은 k: **{k_sil}**",
         f"- 최종 k: **{k}** ({'실루엣 최대' if K_FIXED is None else '직접 지정 — 근거는 3절 비교'})",
         f"- 최종 군집 이름: {', '.join(f'{c} = {n}' for c, n in names.items()) if names else '자동 제안 이름 사용'}",
         *([f"- **[경고] {w}**" for w in name_warnings] or
           (["- 이름 확인: 각 이름이 오전 첨두 도착 비율 조건(평일 오전 출발 집중형 ≤ 0.4, 시간대 분산형 0.4~0.7, "
             "평일 오전 도착 집중형 ≥ 0.7)과 일치"]
            if names else [])), "",
         "## 3. k별 군집 내용 비교", "",
         "실루엣은 '가장 뚜렷하게 갈리는 구분'을 알려 주지만, 그 구분 안에 성격이 다른 대여소가 섞여 있을 수 있다. "
         "k를 늘렸을 때 새로 갈라지는 군집이 해석 가능한지 비교한다.", "",
         to_md(comp_tab), "",
         "- 규칙과의 ARI: 단순 규칙(비근무일형·오전 도착형·오전 출발형·혼합형)과 군집이 얼마나 같은지 (1 = 같음, 0 = 무관)", ""]
    for kk, v in compare.items():
        L += [f"### k = {kk}", "", f"![k={kk}]({v['fig']})", "", to_md(fmt_summary(v["summary"]), index=True), "",
              "군집 × 규칙 교차표 (대여소 수):", "", to_md(v["cross"], index=True), ""]
    L += ["- 비근무일/근무일 이용 비: 비근무일 하루 평균 이용 ÷ 근무일 하루 평균 이용 (1보다 크면 쉬는 날에 더 많이 쓰임)",
          "- 오전·오후 첨두 도착 비율: 근무일 7~9시대·16~18시대의 출발+도착 중 도착의 비율 (0.5보다 크면 도착 우세)", "",
          f"## 4. 최종 군집 (k = {k})", "", f"![군집 프로파일]({f_p})", "", to_md(fmt_summary(summary), index=True), "",
          "## 5. 검증 (최종 k)", "",
          "### 5.1 초기값(시드) 안정성", "",
          f"- 시드 {len(SEEDS)}개로 다시 군집화했을 때 최종 결과와의 ARI: 평균 {np.mean(seed_ari):.3f}, 최소 {np.min(seed_ari):.3f}",
          "- ARI는 두 군집 결과가 같으면 1, 무작위 수준이면 0에 가깝다.", "",
          "### 5.2 연도 안정성", "", to_md(pd.DataFrame(year_rows)), "",
          "- '배정': 3년 합으로 만든 군집 중심에 각 연도의 프로파일을 배정해, 대여소의 유형이 해마다 유지되는지 본다.", "",
          "### 5.3 전처리 기준 민감도", "", to_md(pd.DataFrame(var_rows)), "",
          "### 5.4 단순 규칙과 비교", "",
          f"- 규칙: 비근무일/근무일 이용 비 ≥ {RULE_NONWORK_RATE} → 비근무일형, 아니면 오전 첨두 도착 비율 ≥ {RULE_AM_HIGH} → "
          f"오전 도착형, ≤ {RULE_AM_LOW} → 오전 출발형, 나머지 혼합형",
          f"- 군집 결과와 규칙의 ARI: **{rule_ari:.3f}**", "", "군집 × 규칙 교차표 (대여소 수):", "",
          to_md(cross, index=True), "",
          "## 6. 해석 작성 시 확인할 것", "",
          "- 군집 이름은 그래프의 모양을 보고 확정한다. 같은 제안 이름이 둘 이상이면 규모·오후 패턴 차이로 구분한다.",
          "- 대여소 이름에 Station·Park 등이 들어간 것은 참고만 한다. 좌표가 없어 주변 환경은 확인할 수 없다.",
          "- 규칙과의 ARI가 높으면 '단순 지표 두 개로도 대부분 설명된다', 낮으면 '시간대 모양에 규칙이 놓치는 정보가 있다'는 뜻이다.",
          ""]
    (OUT_DIR / "q1_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[완료] k = {k} (실루엣 최대 {k_sil}) → {OUT_DIR / 'q1_report.md'}")


if __name__ == "__main__":
    main()