"""Win probability: logistic regression trained on real GeBIZ telco/cloud/cyber/IoT awards.

Label = Singtel group (Singtel or NCS) won the tender. Features use only awards dated BEFORE each tender,
so the model never sees the future. Coefficients are stored in kv and applied with plain math at score time.
"""
import math
from collections import defaultdict

from . import db
from .collect import is_singtel

CATS = ["connectivity", "cloud", "cyber", "iot"]
NAMES = ["singtel_incumbent", "singtel_at_agency", "log_amount"] + [f"cat_{c}" for c in CATS]
SPLIT = "2024-07-01"  # train before, test after


def tenders(con):
    """One row per tender: agency, category, date, amount, singtel_won. Ordered by date."""
    t = {}
    for r in con.execute("SELECT * FROM awards ORDER BY award_date"):
        x = t.setdefault(r["tender_no"], dict(tender_no=r["tender_no"], agency=r["agency"], cat=r["category"], date=r["award_date"],
                                               amount=0.0, won=False, suppliers=[]))
        x["amount"] += r["amount"]
        x["won"] |= is_singtel(r["supplier"])
        x["suppliers"].append(r["supplier"])
    return sorted(t.values(), key=lambda x: x["date"])


def vec(incumbent, at_agency, amount, cat):
    return [float(incumbent), float(at_agency), math.log1p(amount) / 10] + [float(cat == c) for c in CATS]


def build(con):
    """Feature rows with history-only features. Returns X, y, dates."""
    last = {}                        # (agency, cat) -> singtel won last time
    seen = defaultdict(bool)         # agency -> singtel ever won there
    X, y, dates = [], [], []
    for t in tenders(con):
        X.append(vec(last.get((t["agency"], t["cat"]), False), seen[t["agency"]], t["amount"], t["cat"]))
        y.append(int(t["won"]))
        dates.append(t["date"])
        last[(t["agency"], t["cat"])] = t["won"]
        seen[t["agency"]] |= t["won"]
    return X, y, dates


def train(con):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, y, dates = build(con)
    tr = [i for i, d in enumerate(dates) if d < SPLIT]
    te = [i for i, d in enumerate(dates) if d >= SPLIT]
    m = LogisticRegression(max_iter=1000).fit([X[i] for i in tr], [y[i] for i in tr])
    p = m.predict_proba([X[i] for i in te])[:, 1]
    auc = roc_auc_score([y[i] for i in te], p)
    final = LogisticRegression(max_iter=1000).fit(X, y)  # refit on everything for live use
    amounts = sorted(x[2] for x in X)
    out = {
        "intercept": float(final.intercept_[0]),
        "coef": dict(zip(NAMES, map(float, final.coef_[0]))),
        "median_log_amount": amounts[len(amounts) // 2],
        "test_auc": round(float(auc), 3),
        "n_train": len(tr), "n_test": len(te),
        "base_rate": round(sum(y) / len(y), 3),
        "split": SPLIT,
    }
    db.kv_set(con, "winprob", out)
    con.commit()
    return out


def predict(m, incumbent, at_agency, cat, amount=None):
    if not m:
        return None
    x = vec(incumbent, at_agency, 0, cat)
    x[2] = math.log1p(amount) / 10 if amount else m["median_log_amount"]
    z = m["intercept"] + sum(c * v for c, v in zip(m["coef"].values(), x))
    return round(1 / (1 + math.exp(-z)), 3)


if __name__ == "__main__":
    fake = {"intercept": -2.0, "coef": dict.fromkeys(NAMES, 0.0) | {"singtel_incumbent": 2.0},
            "median_log_amount": 1.3}
    assert predict(fake, True, True, "cloud") == 0.5
    assert predict(fake, False, False, "cloud") < 0.2
    print("model ok")
