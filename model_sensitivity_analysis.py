"""
model_sensitivity_analysis.py
==============================
Model Sensitivity Analysis for the MOMENT compatibility model.
Understand which demographic features most influence think_R and feel_R scores?

Three analyses:
  1. Correlation Analysis     — how strongly does each feature correlate with scores?
  2. Group Mean Comparison    — average scores per group for each feature
  3. Feature Importance       — which feature explains the most variance in scores?

"""

import argparse
import json
import logging
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
USERS_FILE   = "data/users_processed.json"
PAIRS_FILE   = "data/reports/all_profile_scores_fixed.json"
OUTPUT_DIR   = "data/sensitivity_results"

# Features to analyse
FEATURE_COLS = ["gender", "age_group", "personality", "reader_type"]

# Score columns to analyse sensitivity for
SCORE_COLS = ["think_R", "feel_R"]

# Minimum group size to include in analysis
MIN_GROUP_SIZE = 10


# HELPERS
def to_age_group(age):
    try:
        a = int(age)
        if a < 25: return "18-24 (Gen Z)"
        if a < 35: return "25-34 (Millennial)"
        if a < 45: return "35-44 (Gen X/Mill)"
        return "45+ (Gen X/Boom)"
    except Exception:
        return "Unknown"


# ============================================================
# LOAD DATA
# ============================================================
def load_user_demographics(users_path):
    """Load users_processed.json and return a dict keyed by character_name."""
    logger.info("=" * 60)
    logger.info("LOADING USER DEMOGRAPHICS")
    logger.info("=" * 60)

    if not os.path.exists(users_path):
        logger.warning(f"User data not found at {users_path} — demographics will be skipped.")
        return {}

    with open(users_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = list(data.values())

    demo = {}
    for user in data:
        name = str(user.get("character_name", user.get("name", "")))
        if not name:
            continue
        demo[name] = {
            "gender":      str(user.get("gender",                "Unknown")),
            "age_group":   to_age_group(user.get("age",          "Unknown")),
            "personality": str(user.get("personality",           "Unknown")),
            "reader_type": str(user.get("distribution_category", "Unknown")),
        }

    logger.info(f"Loaded {len(demo)} users from {users_path}")
    return demo


def load_pairs(pairs_path, demo):
    """
    Load all_profile_scores_fixed.json.
    Flattens overall + by_book into one row per (user_a, user_b, book).
    Returns a flat DataFrame.
    """
    logger.info("=" * 60)
    logger.info("LOADING PAIR SCORES")
    logger.info("=" * 60)

    with open(pairs_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("pairs", [])

    rows = []
    for obj in data:
        user_a = str(obj.get("user_a", ""))
        user_b = str(obj.get("user_b", ""))

        d = demo.get(user_a, {})
        base_demo = {
            "gender":      d.get("gender",      "Unknown"),
            "age_group":   d.get("age_group",   "Unknown"),
            "personality": d.get("personality", "Unknown"),
            "reader_type": d.get("reader_type", "Unknown"),
        }

        def make_row(scores, book_name):
            think = scores.get("think", {})
            feel  = scores.get("feel",  {})
            return {
                "user_a":      user_a,
                "user_b":      user_b,
                "book":        book_name,
                "think_R":     think.get("R", 0),
                "think_C":     think.get("C", 0),
                "think_D":     think.get("D", 0),
                "feel_R":      feel.get("R",  0),
                "feel_C":      feel.get("C",  0),
                "feel_D":      feel.get("D",  0),
                "confidence":  scores.get("confidence", 0.0),
                **base_demo,
            }

        if "overall" in obj:
            rows.append(make_row(obj["overall"], "overall"))
        for book_name, book_scores in obj.get("by_book", {}).items():
            rows.append(make_row(book_scores, book_name))

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} rows from {pairs_path}")
    return df


# ============================================================
# ANALYSIS 1 — CORRELATION ANALYSIS
# How strongly does each feature correlate with think_R / feel_R?
# Uses eta-squared (effect size for categorical → continuous)
# ============================================================
def correlation_analysis(df, report):
    """
    Compute eta-squared for each feature vs each score column.
    higher = stronger relationship.

    Interpretation:
      eta² < 0.01  → negligible
      eta² 0.01–0.06 → small
      eta² 0.06–0.14 → medium
      eta² > 0.14  → large
    """
    logger.info("=" * 60)
    logger.info("1. CORRELATION ANALYSIS (Eta-Squared)")
    logger.info("=" * 60)

    report.append(f"\n{'=' * 60}\n")
    report.append("## 1. CORRELATION ANALYSIS\n\n")
    report.append("Eta-squared measures how much variance in each score is explained by each demographic feature.\n\n")
    report.append("| Feature | think_R η² | feel_R η² | think interpretation | feel interpretation |\n")
    report.append("|---|---|---|---|---|\n")

    results = {}

    for feat in FEATURE_COLS:
        if feat not in df.columns:
            continue

        feat_results = {}

        for score in SCORE_COLS:
            # filter unknowns
            mask = df[feat].astype(str).str.lower() != "unknown"
            sub  = df[mask].copy()

            groups = [
                grp[score].values
                for _, grp in sub.groupby(feat)
                if len(grp) >= MIN_GROUP_SIZE
            ]

            if len(groups) < 2:
                feat_results[score] = {"eta_sq": None, "interpretation": "insufficient data"}
                continue

            # one-way ANOVA → eta-squared
            f_stat, p_val = stats.f_oneway(*groups)
            grand_mean    = sub[score].mean()
            ss_between    = sum(
                len(g) * (g.mean() - grand_mean) ** 2
                for g in groups
            )
            ss_total = sum((sub[score] - grand_mean) ** 2)
            eta_sq   = ss_between / ss_total if ss_total > 0 else 0.0

            if eta_sq < 0.01:
                interp = "negligible"
            elif eta_sq < 0.06:
                interp = "small"
            elif eta_sq < 0.14:
                interp = "medium"
            else:
                interp = "large"

            feat_results[score] = {
                "eta_sq":         round(float(eta_sq), 4),
                "f_stat":         round(float(f_stat), 4),
                "p_value":        round(float(p_val),  4),
                "interpretation": interp,
                "significant":    p_val < 0.05,
            }

            logger.info(f"  {feat:<15} → {score}: η²={eta_sq:.4f} ({interp})"
                        f"  p={p_val:.4f} {'*' if p_val < 0.05 else ''}")

        results[feat] = feat_results

        think = feat_results.get("think_R", {})
        feel  = feat_results.get("feel_R",  {})
        report.append(
            f"| {feat} "
            f"| {think.get('eta_sq', 'N/A')} "
            f"| {feel.get('eta_sq',  'N/A')} "
            f"| {think.get('interpretation', 'N/A')} "
            f"| {feel.get('interpretation',  'N/A')} |\n"
        )

    report.append("\n*p < 0.05 indicates statistically significant relationship.\n")
    return results


# ============================================================
# ANALYSIS 2 — GROUP MEAN COMPARISON
# Average think_R and feel_R per group for each feature
# ============================================================
def group_mean_comparison(df, report):
    logger.info("=" * 60)
    logger.info("2. GROUP MEAN COMPARISON")
    logger.info("=" * 60)

    report.append(f"\n{'=' * 60}\n")
    report.append("## 2. GROUP MEAN COMPARISON\n\n")
    report.append("Mean scores per demographic group. ")
    report.append("Flags groups deviating more than 1 std from overall mean.\n")

    results = {}

    for feat in FEATURE_COLS:
        if feat not in df.columns:
            continue

        mask = df[feat].astype(str).str.lower() != "unknown"
        sub  = df[mask].copy()

        logger.info(f"\n  Feature: {feat}")
        report.append(f"\n### {feat}\n\n")
        report.append(f"| Group | Count | mean think_R | mean feel_R | think flag | feel flag |\n")
        report.append(f"|---|---|---|---|---|---|\n")

        feat_results = {}
        overall_think = sub["think_R"].mean()
        overall_feel  = sub["feel_R"].mean()
        std_think     = sub["think_R"].std()
        std_feel      = sub["feel_R"].std()

        logger.info(f"  {'Group':<25} {'Count':>7} {'mean think_R':>14} {'mean feel_R':>12}")
        logger.info(f"  {'-' * 62}")

        for grp_name, grp_df in sub.groupby(feat):
            if len(grp_df) < MIN_GROUP_SIZE:
                continue

            m_think = grp_df["think_R"].mean()
            m_feel  = grp_df["feel_R"].mean()
            s_think = grp_df["think_R"].std()
            s_feel  = grp_df["feel_R"].std()

            think_flag = "⚠" if abs(m_think - overall_think) > std_think else "✓"
            feel_flag  = "⚠" if abs(m_feel  - overall_feel)  > std_feel  else "✓"

            logger.info(f"  {str(grp_name):<25} {len(grp_df):>7} "
                        f"{m_think:>12.1f} {think_flag}  "
                        f"{m_feel:>10.1f} {feel_flag}")

            report.append(
                f"| {grp_name} | {len(grp_df)} "
                f"| {m_think:.1f} | {m_feel:.1f} "
                f"| {think_flag} | {feel_flag} |\n"
            )

            feat_results[str(grp_name)] = {
                "count":        len(grp_df),
                "mean_think_R": round(float(m_think), 2),
                "mean_feel_R":  round(float(m_feel),  2),
                "std_think_R":  round(float(s_think), 2),
                "std_feel_R":   round(float(s_feel),  2),
                "think_flag":   think_flag == "⚠",
                "feel_flag":    feel_flag  == "⚠",
            }

        logger.info(f"  {'-' * 62}")
        logger.info(f"  {'Overall':<25} {len(sub):>7} "
                    f"{overall_think:>12.1f}    {overall_feel:>10.1f}")

        report.append(f"| **Overall** | {len(sub)} "
                      f"| {overall_think:.1f} | {overall_feel:.1f} | - | - |\n")

        results[feat] = {
            "overall_mean_think_R": round(float(overall_think), 2),
            "overall_mean_feel_R":  round(float(overall_feel),  2),
            "groups":               feat_results,
        }

    return results


# ============================================================
# ANALYSIS 3 — FEATURE IMPORTANCE RANKING
# Which feature explains the most variance in think_R / feel_R?
# ============================================================
def feature_importance_ranking(correlation_results, report):
    logger.info("=" * 60)
    logger.info("3. FEATURE IMPORTANCE RANKING")
    logger.info("=" * 60)

    report.append(f"\n{'=' * 60}\n")
    report.append("## 3. FEATURE IMPORTANCE RANKING\n\n")
    report.append("Features ranked by eta-squared.\n\n")
    report.append("Higher eta-squared = more important feature.\n\n")

    results = {}

    for score in SCORE_COLS:
        ranked = sorted(
            [
                (feat, corr.get(score, {}).get("eta_sq", 0) or 0)
                for feat, corr in correlation_results.items()
            ],
            key=lambda x: x[1],
            reverse=True
        )

        logger.info(f"\n  [{score}] Feature importance ranking:")
        report.append(f"### {score}\n\n")
        report.append(f"| Rank | Feature | Eta-squared | Interpretation |\n")
        report.append(f"|---|---|---|---|\n")

        score_results = []
        for rank, (feat, eta_sq) in enumerate(ranked, 1):
            interp = correlation_results.get(feat, {}).get(score, {}).get("interpretation", "N/A")
            logger.info(f"  {rank}. {feat:<15} η²={eta_sq:.4f} ({interp})")
            report.append(f"| {rank} | {feat} | {eta_sq:.4f} | {interp} |\n")
            score_results.append({
                "rank":           rank,
                "feature":        feat,
                "eta_sq":         eta_sq,
                "interpretation": interp,
            })

        results[score] = score_results
        report.append("\n")

    return results


# ============================================================
# SUMMARY
# ============================================================
def write_summary(correlation_results, importance_results, report):
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    report.append(f"\n{'=' * 60}\n")
    report.append("## SUMMARY\n\n")

    for score in SCORE_COLS:
        ranked = importance_results.get(score, [])
        if not ranked:
            continue

        top    = ranked[0]
        bottom = ranked[-1]

        logger.info(f"\n  [{score}]")
        logger.info(f"  Most influential  : {top['feature']} (η²={top['eta_sq']:.4f}, {top['interpretation']})")
        logger.info(f"  Least influential : {bottom['feature']} (η²={bottom['eta_sq']:.4f}, {bottom['interpretation']})")

        report.append(f"**{score}**\n")
        report.append(f"- Most influential feature: `{top['feature']}` "
                      f"(η²={top['eta_sq']:.4f}, {top['interpretation']} effect)\n")
        report.append(f"- Least influential feature: `{bottom['feature']}` "
                      f"(η²={bottom['eta_sq']:.4f}, {bottom['interpretation']} effect)\n\n")

    report.append("### Interpretation guide\n\n")
    report.append("- **Negligible (η² < 0.01)**: feature has almost no influence on scores\n")
    report.append("- **Small (0.01–0.06)**: feature has minor influence\n")
    report.append("- **Medium (0.06–0.14)**: feature has moderate influence\n")
    report.append("- **Large (η² > 0.14)**: feature has strong influence on scores\n\n")
    report.append("A large effect means the model's scores vary significantly across groups "
                  "for that feature — worth investigating whether this reflects genuine "
                  "reader differences or potential model sensitivity.\n")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MOMENT model sensitivity analysis")
    parser.add_argument("--pairs",  default=PAIRS_FILE,  help="Path to pairs output JSON")
    parser.add_argument("--users",  default=USERS_FILE,  help="Path to users_processed.json")
    parser.add_argument("--output", default=OUTPUT_DIR,  help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MODEL SENSITIVITY ANALYSIS — MOMENT PROJECT")
    logger.info("=" * 60)

    # ── Load ──
    demo = load_user_demographics(args.users)
    df   = load_pairs(args.pairs, demo)
    logger.info(f"Total rows for analysis: {len(df)}")

    # ── Report scaffold ──
    report = []
    report.append("# MODEL SENSITIVITY ANALYSIS REPORT — MOMENT PROJECT\n\n")
    report.append(f"Date: {pd.Timestamp.now().date()}\n")
    report.append(f"Total rows analysed: {len(df)}\n")
    report.append(f"Features analysed: {', '.join(FEATURE_COLS)}\n")
    report.append(f"Scores analysed: {', '.join(SCORE_COLS)}\n")

    # ── Analysis 1: Correlation ──
    corr_results = correlation_analysis(df, report)

    # ── Analysis 2: Group means ──
    group_results = group_mean_comparison(df, report)

    # ── Analysis 3: Feature importance ──
    importance_results = feature_importance_ranking(corr_results, report)

    # ── Summary ──
    write_summary(corr_results, importance_results, report)

    # ── Save report ──
    report_path = os.path.join(args.output, "sensitivity_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    logger.info(f"\n✓ Report  → {report_path}")

    # ── Save results JSON ──
    results = {
        "correlation":  corr_results,
        "group_means":  group_results,
        "importance":   importance_results,
    }
    results_path = os.path.join(args.output, "sensitivity_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: bool(o) if isinstance(o, (bool, np.bool_)) else str(o))
    logger.info(f"✓ Results → {results_path}")

    logger.info(f"✓ Done. All outputs in: {args.output}/")


if __name__ == "__main__":
    main()