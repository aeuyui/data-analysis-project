"""
3차 3단계 — 질문 2: 이용이 많은 흐름은 무엇이며, 출발과 도착이 크게 불균형한 대여소는 어디인가?

정의
- 불균형 지수 I = (도착 − 출발) / (도착 + 출발), 범위 −1 ~ +1
    +에 가까울수록 도착 우세(자전거가 쌓임), −에 가까울수록 출발 우세(자전거가 빠짐)
- 순유입 = (도착 − 출발) / 날짜 수  → 하루 평균 몇 대가 쌓이거나 빠지는지 (규모)
- 시간대별로 따로 계산한다: 평일 오전 첨두(7~9시대), 평일 오후 첨두(16~18시대), 평일 하루 전체, 주말·공휴일 하루 전체
  (EDA에서 오전과 오후의 방향이 반대로 나타나, 하루로 합치면 상쇄될 수 있기 때문)
- '크게 불균형' = 이용이 충분한 대여소(시간대 하루 평균 출발+도착 3건 이상) 중 한쪽이 다른 쪽의 3배 이상(|I| ≥ 0.5)
  → 2배·4배 기준의 결과도 함께 기록하고, 분포(중앙값, 90% 지점)를 보고서에 남긴다.

검증
1. 상쇄 확인: 오전에 크게 불균형한 대여소가 하루 전체로는 균형에 가까운가, 오전·오후 순유입의 상관
2. 연도 일관성: 연도별 순유입 순위 상관(스피어만), 상위 10개 대여소가 해마다 겹치는 정도
3. 전처리 기준 민감도: 기준을 바꾼 데이터에서도 순위가 유지되는가
4. 질문 1의 대여소 유형과 연결

입력: data/agg/station_hourly_year.parquet, data/agg/od.parquet (03_eda.py), data/agg/station_clusters.csv (있으면)
출력: data/agg/station_imbalance.csv, 03_analysis-modeling/q2_report.md, figures/q2_*.png

사용법 (proj 폴더에서 실행)
    python 06_q2_imbalance.py
"""

from datetime import datetime

import numpy as np
import pandas as pd

from common import (AGG_DIR, AM_PEAK, C, OUT_DIR, PM_PEAK, day_counts, finish, fmt_int, load_station_hourly,
                    new_ax, setup_font, to_md)

BANDS = {
    "평일 오전 첨두(7~9시대)": (True, AM_PEAK),
    "평일 오후 첨두(16~18시대)": (True, PM_PEAK),
    "평일 하루 전체": (True, list(range(24))),
    "주말·공휴일 하루 전체": (False, list(range(24))),
}
AM, PM, WDAY, NWDAY = list(BANDS)
# 시간대 출발+도착이 하루 평균 3건 이상: 3년 평일(약 750일)이면 2,000건 이상이라 지수가 우연에 크게 흔들리지 않는다.
MIN_BAND_PER_DAY = 3
# '크게 불균형' = 한쪽이 다른 쪽의 3배 이상. 도착 = r × 출발이면 I = (r − 1) / (r + 1) → r = 3일 때 |I| = 0.5
STRONG_RATIO = 3
RATIO_CHECKS = [2, 3, 4]       # 기준 민감도: 2배·3배·4배
TOP_N = 10


def ratio_to_index(r: float) -> float:
    return (r - 1) / (r + 1)


def band_table(sh: pd.DataFrame, years=None) -> pd.DataFrame:
    dc = day_counts()
    if years is not None:
        sh = sh[sh["year"].isin(list(years))]
        dc = dc[dc["year"].isin(list(years))]
    days = dc.groupby("workday")["days"].sum()
    parts = []
    for band, (wd, hours) in BANDS.items():
        g = sh[(sh["workday"] == wd) & sh["hour"].isin(hours)].groupby("station_key")[["starts", "ends"]].sum()
        d = days.get(wd, np.nan)
        g = g.assign(band=band,
                     starts_per_day=g["starts"] / d, ends_per_day=g["ends"] / d,
                     volume_per_day=(g["starts"] + g["ends"]) / d,
                     net_per_day=(g["ends"] - g["starts"]) / d,
                     index=(g["ends"] - g["starts"]) / (g["ends"] + g["starts"]).where(lambda s: s > 0))
        parts.append(g.reset_index())
    t = pd.concat(parts, ignore_index=True)
    t["eligible"] = t["volume_per_day"] >= MIN_BAND_PER_DAY
    return t


def spearman(a: pd.Series, b: pd.Series) -> float:
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    return j.iloc[:, 0].rank().corr(j.iloc[:, 1].rank()) if len(j) > 2 else np.nan


def top_sets(s: pd.Series, n=TOP_N):
    return set(s.nlargest(n).index), set(s.nsmallest(n).index)


def main():
    setup_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sh = load_station_hourly("base")
    years = sorted(sh["year"].unique())
    t = band_table(sh)

    # '크게 불균형' 기준 (모든 시간대에 같은 기준)
    thr = {band: ratio_to_index(STRONG_RATIO) for band in BANDS}
    t["threshold"] = t["band"].map(thr)
    t["strong"] = t["eligible"] & (t["index"].abs() >= t["threshold"])
    t["direction"] = np.where(~t["eligible"], "이용 적음(판단 제외)",
                              np.where(~t["strong"], "균형~보통",
                                       np.where(t["index"] > 0, "도착 우세", "출발 우세")))

    def piv(col):
        return t.pivot(index="station_key", columns="band", values=col)

    idx_w, net_w = piv("index").astype(float), piv("net_per_day").astype(float)
    elig_w = piv("eligible").fillna(False).astype(bool)

    # 1. 상쇄 확인
    am_strong = t[(t["band"] == AM) & t["strong"]]["station_key"]
    day_idx = idx_w[WDAY].reindex(am_strong)
    both = elig_w[AM] & elig_w[PM]
    am_net, pm_net = net_w[AM][both], net_w[PM][both]
    corr_p = am_net.corr(pm_net)
    corr_s = spearman(am_net, pm_net)

    # 2. 연도 일관성 (오전·오후 첨두 순유입)
    yt = {y: band_table(sh, years=[y]) for y in years}
    year_rows = []
    for band in [AM, PM]:
        nets = {}
        for y in years:
            yy = yt[y]
            yy = yy[(yy["band"] == band) & yy["eligible"]].set_index("station_key")["net_per_day"]
            nets[y] = yy
        for a, b in zip(years, years[1:]):
            ia, oa = top_sets(nets[a])
            ib, ob = top_sets(nets[b])
            year_rows.append({"시간대": band, "비교": f"{a} vs {b}", "순유입 순위 상관(스피어만)": round(spearman(nets[a], nets[b]), 3),
                              f"도착 상위 {TOP_N}개 겹침": f"{len(ia & ib)}/{TOP_N}", f"출발 상위 {TOP_N}개 겹침": f"{len(oa & ob)}/{TOP_N}"})

    # 3. 전처리 기준 민감도
    var_rows = []
    base_am = t[(t["band"] == AM) & t["eligible"]].set_index("station_key")["net_per_day"]
    bi, bo = top_sets(base_am)
    for v, desc in [("short2", "짧은 왕복 2분 이하 추가 제외"), ("long3h", "3시간 초과 추가 제외")]:
        tv = band_table(load_station_hourly(v))
        va = tv[(tv["band"] == AM) & tv["eligible"]].set_index("station_key")["net_per_day"]
        vi, vo = top_sets(va)
        var_rows.append({"변형": desc, "오전 첨두 순유입 순위 상관": round(spearman(base_am, va), 4),
                         f"도착 상위 {TOP_N}개 겹침": f"{len(bi & vi)}/{TOP_N}", f"출발 상위 {TOP_N}개 겹침": f"{len(bo & vo)}/{TOP_N}"})

    # 4. 질문 1 유형과 연결
    cpath = AGG_DIR / "station_clusters.csv"
    cross = None
    if cpath.exists():
        cl = pd.read_csv(cpath).set_index("station_key")
        name_col = "cluster_name" if "cluster_name" in cl.columns else "suggested_label"
        am_dir = t[t["band"] == AM].set_index("station_key")["direction"]
        j = am_dir.to_frame("오전 첨두 불균형").join(cl[["cluster", name_col]], how="inner")
        j["유형"] = j["cluster"].astype(str) + ". " + j[name_col]
        cross = pd.crosstab(j["유형"], j["오전 첨두 불균형"])

    # 주요 흐름 (03_eda.py의 od 집계, 운영 장애일 184건 포함 — 영향 무시 가능)
    od = pd.read_parquet(AGG_DIR / "od.parquet")
    dc = day_counts()
    n_work = int(dc.loc[dc["workday"], "days"].sum())
    od_band_name = {AM: "평일 오전 첨두(7~9시대)", PM: "평일 오후 첨두(16~18시대)"}
    flows = {}
    for band, name in od_band_name.items():
        f = od[(od["band"] == name) & (od["start_station_key"] != od["end_station_key"])]
        f = f.nlargest(TOP_N, "trips").assign(per_workday=lambda d: (d["trips"] / n_work).round(1))
        flows[band] = f[["start_station_key", "end_station_key", "trips", "per_workday"]]
    worst = base_am.idxmin()
    from_worst = od[(od["band"] == od_band_name[AM]) & (od["start_station_key"] == worst)
                    & (od["end_station_key"] != worst)]
    dest_share = from_worst.nlargest(5, "trips").assign(
        share=lambda d: (d["trips"] / from_worst["trips"].sum() * 100).round(1))

    # 저장
    t.to_csv(AGG_DIR / "station_imbalance.csv", index=False, encoding="utf-8-sig")

    # 그래프 1: 오전 vs 오후 순유입
    fig, ax = new_ax(7.5, 6)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.scatter(am_net, pm_net, s=14, color=C["s1"], alpha=0.6, edgecolors=C["surface"], linewidths=0.5)
    lim = np.nanmax(np.abs(np.r_[am_net.values, pm_net.values])) * 1.1
    ax.plot([-lim, lim], [lim, -lim], color=C["muted"], linewidth=1, linestyle="--")
    ax.axhline(0, color=C["axis"], linewidth=0.8)
    ax.axvline(0, color=C["axis"], linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    # 오전 순유입이 큰 대여소 6곳에 이름표. 가까운 점끼리 겹치지 않도록 세로로 엇갈려 배치
    for i, name in enumerate(am_net.reindex(am_net.abs().nlargest(6).index).sort_values().index):
        dy = [12, -14, 26, -28, 40, -42][i]
        ax.annotate(name.split(",")[0], (am_net[name], pm_net[name]), xytext=(8, dy), textcoords="offset points",
                    fontsize=8, color=C["ink2"], arrowprops=dict(arrowstyle="-", color=C["axis"], linewidth=0.6))
    ax.set_xlabel("평일 오전 첨두 순유입 (도착 - 출발, 하루 평균)", color=C["ink2"], fontsize=9)
    ax.set_ylabel("평일 오후 첨두 순유입 (하루 평균)", color=C["ink2"], fontsize=9)
    f_sc = finish(fig, "오전과 오후의 순유입은 반대 방향인가",
                  f"점 = 대여소 {len(am_net)}개, 점선 = 오전과 오후가 정확히 반대인 경우 (스피어만 {corr_s:.2f})",
                  "q2_am_pm_scatter.png")

    # 그래프 2: 불균형 지수 분포 (오전·오후)
    fig, axes = new_ax(10, 3.6, ncols=2, sharey=True)
    for ax, band in zip(axes, [AM, PM]):
        e = t[(t["band"] == band) & t["eligible"]]["index"]
        ax.hist(e, bins=np.linspace(-1, 1, 41), color=C["s1"], edgecolor=C["surface"], linewidth=0.8)
        for s in (-1, 1):
            ax.axvline(s * thr[band], color=C["ink2"], linewidth=1, linestyle="--")
        ax.set_title(f"{band}  (기준 |I| ≥ {thr[band]:.2f})", loc="left", fontsize=9, color=C["ink2"])
        ax.set_xlabel("불균형 지수 I  (−1 출발만 ~ +1 도착만)".replace("−", "-"), color=C["ink2"], fontsize=9)
    axes[0].set_ylabel("대여소 수", color=C["ink2"], fontsize=9)
    f_hist = finish(fig, "불균형 지수 분포", f"점선 바깥 = 크게 불균형 (한쪽이 다른 쪽의 {STRONG_RATIO}배 이상, |I| ≥ "
                    f"{ratio_to_index(STRONG_RATIO):.2f})", "q2_index_distribution.png")

    # 그래프 3: 연도 일관성 (첫 해 vs 마지막 해, 오전 첨두)
    fig, ax = new_ax(6.5, 5.6)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    a = yt[years[0]].query("band == @AM and eligible").set_index("station_key")["net_per_day"]
    b = yt[years[-1]].query("band == @AM and eligible").set_index("station_key")["net_per_day"]
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    ax.scatter(j.iloc[:, 0], j.iloc[:, 1], s=14, color=C["s1"], alpha=0.6, edgecolors=C["surface"], linewidths=0.5)
    lim = np.abs(j.to_numpy()).max() * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=C["muted"], linewidth=1, linestyle="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"{years[0]}년 오전 첨두 순유입 (하루 평균)", color=C["ink2"], fontsize=9)
    ax.set_ylabel(f"{years[-1]}년 오전 첨두 순유입 (하루 평균)", color=C["ink2"], fontsize=9)
    f_yr = finish(fig, "오전 첨두 불균형은 해마다 같은 곳에서 나타나는가",
                  f"점선 = 두 해가 같은 경우, 스피어만 {spearman(j.iloc[:, 0], j.iloc[:, 1]):.2f}", "q2_year_consistency.png")

    # 보고서
    def rank_table(band, ascending):
        e = t[(t["band"] == band) & t["eligible"]]
        e = e[e["net_per_day"] < 0] if ascending else e[e["net_per_day"] > 0]  # 방향이 맞는 대여소만
        e = e.sort_values("net_per_day", ascending=ascending).head(TOP_N)
        return e[["station_key", "starts_per_day", "ends_per_day", "net_per_day", "index", "direction"]].round(
            {"starts_per_day": 1, "ends_per_day": 1, "net_per_day": 1, "index": 2}).rename(columns={
            "station_key": "대여소", "starts_per_day": "출발/일", "ends_per_day": "도착/일", "net_per_day": "순유입/일",
            "index": "지수 I", "direction": "판정"})

    thr_tab = pd.DataFrame([{
        "시간대": band,
        "판단 대상 대여소": int(t[(t["band"] == band) & t["eligible"]].shape[0]),
        "도착 우세": int(((t["band"] == band) & (t["direction"] == "도착 우세")).sum()),
        "출발 우세": int(((t["band"] == band) & (t["direction"] == "출발 우세")).sum()),
        "|I| 중앙값": round(t[(t["band"] == band) & t["eligible"]]["index"].abs().median(), 3),
        "|I| 90% 지점": round(t[(t["band"] == band) & t["eligible"]]["index"].abs().quantile(0.9), 3),
    } for band in BANDS])
    sens_tab = pd.DataFrame([{
        "시간대": band,
        **{f"{r}배 이상 (|I| ≥ {ratio_to_index(r):.2f})": int(
            (t[(t["band"] == band) & t["eligible"]]["index"].abs() >= ratio_to_index(r)).sum()) for r in RATIO_CHECKS},
    } for band in BANDS])

    L = ["# 질문 2. 출발-도착 흐름과 불균형", "",
         f"- 실행 일시: {datetime.now():%Y-%m-%d %H:%M}",
         "- 질문: 출발-도착 대여소 쌍 중 이용이 많은 흐름은 무엇이며, 출발과 도착이 크게 불균형한 대여소는 어디인가?",
         "- 운영 장애일(2025-08-05) 제외. 이 보고서는 **관찰(수치)**을 기록한다.", "",
         "## 1. 정의", "",
         "- 불균형 지수 I = (도착 − 출발) / (도착 + 출발). +1에 가까우면 도착만, −1에 가까우면 출발만 있음",
         "- 순유입/일 = (도착 − 출발) / 해당 날짜 수. 하루 평균 쌓이거나(+) 빠지는(−) 자전거 수",
         f"- 판단 대상: 해당 시간대 출발+도착이 하루 평균 {MIN_BAND_PER_DAY}건 이상인 대여소 "
         "(3년 평일 약 750일이면 2,000건 이상이라 지수가 우연에 크게 흔들리지 않음)",
         f"- **크게 불균형**: 한쪽이 다른 쪽의 **{STRONG_RATIO}배 이상** (|I| ≥ {ratio_to_index(STRONG_RATIO):.2f}). "
         "분포 순위(상위 몇 %)보다 뜻이 분명하고, 값이 같은 대여소가 많아도 흔들리지 않는다.", "",
         to_md(thr_tab), "", "기준을 바꿨을 때 '크게 불균형' 대여소 수:", "", to_md(sens_tab), "",
         "## 2. 불균형 대여소", "", f"![지수 분포]({f_hist})", ""]
    for band in [AM, PM]:
        out_t, in_t = rank_table(band, True), rank_table(band, False)
        L += [f"### {band}", "", f"출발이 더 많은 대여소 상위 {TOP_N} (순유입이 가장 작은 순):", "",
              to_md(out_t) if len(out_t) else "- 해당 없음", "",
              f"도착이 더 많은 대여소 상위 {TOP_N}:", "", to_md(in_t) if len(in_t) else "- 해당 없음", "",
              "- '판정' 열이 '균형~보통'이면 순유입 규모는 커도 지수가 기준값에 못 미친 경우다.", ""]
    L += ["## 3. 오전·오후 상쇄", "", f"![오전 vs 오후]({f_sc})", "",
          f"- 오전·오후 첨두 모두 판단 대상인 대여소 {len(am_net)}개에서 오전 순유입과 오후 순유입의 상관: "
          f"피어슨 {corr_p:.3f}, 스피어만 {corr_s:.3f}",
          f"- 오전 첨두에 크게 불균형한 대여소 {len(am_strong)}개의 **평일 하루 전체** 불균형 지수: "
          f"|I| 중앙값 {day_idx.abs().median():.3f}, |I| < 0.1인 대여소 {int((day_idx.abs() < 0.1).sum())}개 "
          f"({(day_idx.abs() < 0.1).mean() * 100:.0f}%)",
          "",
          "## 4. 검증", "", "### 4.1 연도 일관성", "", f"![연도 일관성]({f_yr})", "", to_md(pd.DataFrame(year_rows)), "",
          "### 4.2 전처리 기준 민감도", "", to_md(pd.DataFrame(var_rows)), ""]
    if cross is not None:
        L += ["### 4.3 질문 1 대여소 유형과의 관계 (오전 첨두 판정, 대여소 수)", "", to_md(cross, index=True), ""]
    L += ["## 5. 주요 흐름", "", "출발-도착 대여소가 다른 흐름만 (평일 하루 평균 = 3년 합계 ÷ 평일 수):", ""]
    for band, f in flows.items():
        L += [f"### {band}", "", to_md(f.rename(columns={"start_station_key": "출발", "end_station_key": "도착",
                                                        "trips": "3년 합계", "per_workday": "평일 하루 평균"})), ""]
    L += [f"오전 첨두 출발 우세 1위 `{worst}`에서 출발한 대여의 도착지 상위 5 (이 대여소 오전 출발 중 비중):", "",
          to_md(dest_share[["end_station_key", "trips", "share"]].rename(columns={
              "end_station_key": "도착", "trips": "3년 합계", "share": "비중(%)"})), "",
          f"- 이 대여소의 오전 첨두 출발 도착지 수: {from_worst['end_station_key'].nunique():,}곳", "",
          "## 6. 해석 작성 시 확인할 것", "",
          "- 불균형은 '이용 기록'의 차이다. 재고 정보가 없어 실제 자전거 부족·반납 불가가 있었는지는 알 수 없다.",
          "- TfL의 재배치(트럭 이동)는 기록에 없다. 순유입은 이용자 이동만 반영한다.",
          "- 대여소 이름(역, 업무지구)으로 해석할 때는 '이름에 근거한 추정'임을 밝힌다.", ""]
    (OUT_DIR / "q2_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[완료] {OUT_DIR / 'q2_report.md'}")


if __name__ == "__main__":
    main()
