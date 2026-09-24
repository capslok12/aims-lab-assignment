"""
Task 2 - Chi-square feature screening on test-dataset.xlsx
Reproduces every number reported in the assignment.
Usage:  python task2_analysis.py [path/to/test-dataset.xlsx]
Needs:  pandas, numpy, scipy, scikit-learn, openpyxl
"""
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
PATH = sys.argv[1] if len(sys.argv) > 1 else "test-dataset.xlsx"
d = pd.read_excel(PATH)

# ---------------------------------------------------------------- 1. Prep
d["age_group"] = pd.cut(d.age, [-1, 17, 29, 44, 59, 200],
                        labels=["<18", "18-29", "30-44", "45-59", "60+"])
d["disability"] = (d.disabilities_name.astype(str) != "0").astype(int)
d["htn"] = d.profile_hypertensive.astype(int)   # target 1
d["dm"] = d.diabetic.astype(int)                # target 2
d["cvd"] = d.has_cardiovascular_disease
d["stroke"] = d.had_stroke

print("=== 1. Data overview ===")
print(f"rows={len(d)}, households={d.household_id.nunique()}, duplicate user_id={d.user_id.duplicated().sum()}")
print(f"HTN cases={d.htn.sum()} ({d.htn.mean():.2%}), DM cases={d.dm.sum()} ({d.dm.mean():.2%}), "
      f"CVD={d.cvd.sum()}, stroke={d.stroke.sum()}")
print("is_poor unique values:", d.is_poor.unique(), "-> zero variance, excluded")
for c in ["BMI", "SUGAR", "SPO2", "MUAC"]:
    print(f"  {c}: measured for {d[c].notna().mean():.1%}")
print(f"age>100: {(d.age > 100).sum()}, age==0: {(d.age == 0).sum()}, min height: {d.HEIGHT.min():.1f}, "
      f"BMI range: {d.BMI.min():.1f}-{d.BMI.max():.1f}")

# ---------------------------------------------------------------- 2. Chi-square
def chi_square(x, y):
    """Pearson chi-square test of independence on complete cases."""
    m = pd.concat([x, y], axis=1).dropna()
    m.columns = ["x", "y"]
    table = pd.crosstab(m.x, m.y)
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)  # E_ij = R_i*C_j/N
    n = table.values.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))                          # Cramer's V
    return dict(n=n, chi2=chi2, dof=dof, p=p, V=v,
                min_expected=expected.min(), pct_cells_lt5=(expected < 5).mean() * 100)

def benjamini_hochberg(p):
    """BH-adjusted q-values: q_(i) = min_{k>=i} (m/k) * p_(k)."""
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, idx in zip(range(m, 0, -1), order[::-1]):
        prev = min(prev, p[idx] * m / rank)
        q[idx] = prev
    return q

features = {
    "union_name": "union_name", "age_group": "age_group", "total_income": "total_income",
    "gender": "gender", "disability": "disability", "is_freedom_fighter": "is_freedom_fighter",
    "had_stroke": "stroke", "has_cardiovascular_disease": "cvd",
    "RESULT_STAT_BP": "RESULT_STAT_BP", "RESULT_STAT_BMI": "RESULT_STAT_BMI",
    "RESULT_STAT_SUGAR": "RESULT_STAT_SUGAR", "TAG_NAME": "TAG_NAME",
    "RESULT_STAT_PR": "RESULT_STAT_PR", "RESULT_STAT_SPO2": "RESULT_STAT_SPO2",
    "diabetic": "dm", "profile_hypertensive": "htn",
}
targets = {"Hypertension": ("htn", "profile_hypertensive"), "Diabetes": ("dm", "diabetic")}

print("\n=== 2. Chi-square results (Table in Section 2.3) ===")
all_rows = []
for tname, (tcol, tfeature) in targets.items():
    rows = []
    for fname, col in features.items():
        if fname == tfeature:          # do not test a variable against itself
            continue
        r = chi_square(d[col], d[tcol])
        r.update(target=tname, feature=fname)
        rows.append(r)
    res = pd.DataFrame(rows)
    res["q_BH"] = benjamini_hochberg(res.p.values)
    res["significant"] = res.q_BH < 0.05
    all_rows.append(res)
    print(f"\n--- Target: {tname} ---")
    print(res[["feature", "n", "chi2", "dof", "p", "V", "q_BH", "significant", "min_expected"]]
          .sort_values("V", ascending=False).round(4).to_string(index=False))
pd.concat(all_rows).to_csv("chi_square_results.csv", index=False)

# ---------------------------------------------------------------- 3. Fisher (sparse 2x2)
print("\n=== 3. Fisher's exact test for sparse 2x2 tables ===")
for tcol in ["htn", "dm"]:
    for col in ["disability", "is_freedom_fighter", "stroke", "cvd"]:
        odds, p = stats.fisher_exact(pd.crosstab(d[col], d[tcol]).values)
        print(f"{tcol:>3} x {col:<18} OR={odds:8.2f}  p={p:.3g}")
odds, p = stats.fisher_exact(pd.crosstab(d.htn, d.dm).values)
print(f"htn x dm                  OR={odds:8.2f}  p={p:.3g}")
odds, p = stats.fisher_exact(pd.crosstab(d.stroke, d.cvd).values)
print(f"stroke x cvd              OR={odds:8.2f}  p={p:.3g}")
print(f"DM among hypertensive: {d[d.htn == 1].dm.mean():.1%};  among non-hypertensive: {d[d.htn == 0].dm.mean():.1%}")

# ---------------------------------------------------------------- 4. Continuous variables
print("\n=== 4. Mann-Whitney U (continuous variables) ===")
for tname, (tcol, _) in targets.items():
    for c in ["age", "SYSTOLIC", "DIASTOLIC", "BMI", "SUGAR", "PULSE_RATE", "SPO2", "HEIGHT", "WEIGHT"]:
        m = d[[c, tcol]].dropna()
        a, b = m[m[tcol] == 1][c], m[m[tcol] == 0][c]
        u = stats.mannwhitneyu(a, b)
        print(f"{tname:<12} {c:<11} n={len(m):>6}  median(case)={a.median():7.2f}  "
              f"median(non-case)={b.median():7.2f}  p={u.pvalue:.3g}")

# ---------------------------------------------------------------- 5. Analytic outcomes
print("\n=== 5. Prevalence and screening gaps ===")
print((d.groupby("age_group", observed=True)[["htn", "dm"]].mean() * 100).round(2))
print((d.groupby("union_name")[["htn", "dm"]].mean() * 100).round(1).sort_values("htn", ascending=False))
high_bp = d.RESULT_STAT_BP.isin(["Mild High", "Moderate High", "Severe High"]) & (d.htn == 0)
print("High BP reading but not flagged hypertensive:", int(high_bp.sum()))
print("Sugar status 'DIABETIC (need confirmation)' but not flagged diabetic:",
      int(((d.RESULT_STAT_SUGAR == "DIABETIC (need confirmation)") & (d.dm == 0)).sum()))

print("\n=== 6. Indicative predictive models (30% stratified hold-out) ===")
base = pd.get_dummies(d[["age", "gender", "total_income", "union_name"]], drop_first=True).astype(float)
vit = d[["SYSTOLIC", "DIASTOLIC", "PULSE_RATE"]]
sets = {"demographics": base, "demographics+vitals": pd.concat([base, vit.fillna(vit.median())], axis=1)}
for tcol in ["htn", "dm"]:
    for name, X in sets.items():
        Xtr, Xte, ytr, yte = train_test_split(X, d[tcol], test_size=0.3, random_state=42, stratify=d[tcol])
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit((Xtr - mu) / sd, ytr)
        gb = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
        print(f"{tcol:>3} | {name:<20} LR AUC={roc_auc_score(yte, lr.predict_proba((Xte - mu) / sd)[:, 1]):.3f}"
              f"   GB AUC={roc_auc_score(yte, gb.predict_proba(Xte)[:, 1]):.3f}")
