"""How we prove it works: tagger accuracy, GeBIZ renewal back-test, press-release back-test."""
import csv
import statistics
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from agents import llm
from . import collect, db, model
from .catalog import RADAR_OPEN, RULES
from .engine import months_between

EVAL = db.ROOT / "eval"
BT = EVAL  # press-release wins live beside the other measurement data


# ---------------- tagger accuracy ----------------
def export_for_labelling(con, n=300):
    """Random sample of collected items for humans to label (gold_signal column left blank on purpose)."""
    rows = con.execute("SELECT id, title, body, lang, country, published FROM items ORDER BY random() LIMIT ?", (n,))
    out = EVAL / "to_label.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "body", "lang", "country", "published", "gold_signal", "gold_company"])
        for r in rows:
            w.writerow([*r, "", ""])
    return out


def eval_tagger(con, path=None):
    path = Path(path) if path else (EVAL / "labels.csv" if (EVAL / "labels.csv").exists() else EVAL / "dev_seed.csv")
    gold = [r for r in csv.DictReader(path.open()) if r["gold_signal"]]
    for i, g in enumerate(gold):
        g["id"] = i
        g.setdefault("body", "")
    pred = {t.id: t for t in llm.tag(gold)}
    ok = sum(pred[g["id"]].signal == g["gold_signal"] for g in gold)
    rel_tp = sum(pred[g["id"]].signal != "none" and g["gold_signal"] != "none" for g in gold)
    rel_pred = sum(pred[g["id"]].signal != "none" for g in gold)
    rel_gold = sum(g["gold_signal"] != "none" for g in gold)
    co = [g for g in gold if g.get("gold_company")]
    co_ok = sum(db.key(pred[g["id"]].company) == db.key(g["gold_company"]) for g in co)
    errors = Counter((g["gold_signal"], pred[g["id"]].signal) for g in gold if pred[g["id"]].signal != g["gold_signal"])
    per = {}
    for s in sorted({g["gold_signal"] for g in gold}):
        tp = sum(g["gold_signal"] == s and pred[g["id"]].signal == s for g in gold)
        fp = sum(g["gold_signal"] != s and pred[g["id"]].signal == s for g in gold)
        fn = sum(g["gold_signal"] == s and pred[g["id"]].signal != s for g in gold)
        per[s] = {"precision": round(tp / (tp + fp), 2) if tp + fp else None,
                  "recall": round(tp / (tp + fn), 2) if tp + fn else None, "n": tp + fn}
    out = {"file": path.name, "n": len(gold), "accuracy": round(ok / len(gold), 3),
           "relevance_precision": round(rel_tp / rel_pred, 3) if rel_pred else None,
           "relevance_recall": round(rel_tp / rel_gold, 3) if rel_gold else None,
           "company_accuracy": round(co_ok / len(co), 3) if co else None,
           "per_class": per, "top_errors": [[g, p, n] for (g, p), n in errors.most_common(8)],
           "model": llm.MODEL, "at": date.today().isoformat()}
    db.kv_set(con, "tagger_eval", out)
    con.commit()
    return out


# ---------------- GeBIZ renewal back-test ----------------
def backtest_gebiz(con, since="2023-01-01"):
    """For every tender the Singtel group won since `since`: find the earlier contract it renewed (same agency,
    similar description). Would the renewal radar (from catalog.RADAR_OPEN to 3 months before that contract's
    estimated end) have flagged it before the award, and how early? Uses only data dated before each award."""
    awards = con.execute("SELECT * FROM awards ORDER BY award_date").fetchall()
    data_end = max(a["award_date"] for a in awards)
    by_agency = defaultdict(list)
    for a in awards:
        by_agency[a["agency"]].append(a)
    desc = {a["tender_no"]: a["description"] for a in awards}

    rows = []
    for t in model.tenders(con):
        if not t["won"] or t["date"] < since:
            continue
        d = desc[t["tender_no"]]
        cands = [(collect.similar(d, a["description"]), a) for a in by_agency[t["agency"]] if a["award_date"] < t["date"]]
        cands = [x for x in cands if x[0] >= collect.RENEWAL_SIM]
        if not cands:
            rows.append({**t, "renewal": False, "flagged": False})
            continue
        p = max(cands, key=lambda x: (x[0], x[1]["award_date"]))[1]
        end, wd = date.fromisoformat(p["est_end"]), date.fromisoformat(t["date"])
        first = max(date.fromisoformat(p["award_date"]), collect.add_months(end, -RADAR_OPEN))
        flagged = first <= min(collect.add_months(end, -3), wd)
        rows.append({**t, "renewal": True, "flagged": flagged, "incumbent": p["supplier"],
                     "lead_months": round(months_between(first, wd), 1) if flagged else None})
    ren = [r for r in rows if r["renewal"]]
    hit = [r for r in ren if r["flagged"]]
    leads = [r["lead_months"] for r in hit]

    # radar precision: contracts whose radar window closed inside the data; did a similar award follow in the window?
    tp = n = 0
    for lst in by_agency.values():
        for i, p in enumerate(lst):
            end = date.fromisoformat(p["est_end"])
            if collect.add_months(end, 6).isoformat() > data_end:
                continue
            n += 1
            lo, hi = collect.add_months(end, -RADAR_OPEN).isoformat(), collect.add_months(end, 6).isoformat()
            tp += any(lo <= q["award_date"] <= hi and q["tender_no"] != p["tender_no"]
                      and collect.similar(p["description"], q["description"]) >= collect.RENEWAL_SIM for q in lst[i + 1:])
    out = {
        "since": since, "data_end": data_end,
        "singtel_wins": len(rows), "renewal_wins": len(ren), "flagged": len(hit),
        "recall_on_renewals": round(len(hit) / len(ren), 3) if ren else None,
        "recall_on_all_wins": round(len(hit) / len(rows), 3) if rows else None,
        "median_lead_months": statistics.median(leads) if leads else None,
        "defend": sum(collect.is_singtel(r["incumbent"]) for r in hit),
        "attack": sum(not collect.is_singtel(r["incumbent"]) for r in hit),
        "radar_precision": round(tp / n, 3) if n else None, "radar_flags_checked": n,
        "examples": [{k: r.get(k) for k in ("date", "agency", "cat", "incumbent", "lead_months")} for r in hit[:12]],
    }
    db.kv_set(con, "backtest_gebiz", out)
    con.commit()
    return out


# ---------------- press-release back-test ----------------
WIN_QUERIES = ['Singtel enterprise customer selects OR signs OR deploys OR partners',
               'Singtel awarded contract', 'NCS awarded contract', 'Singtel 5G enterprise deployment',
               'Singtel managed services deal']


def harvest_wins(con, start=2021, end=2026):
    """Find candidate past Singtel enterprise wins from dated headlines. Humans verify before trusting."""
    items = []
    for y in range(start, end + 1):
        for h in ((f"{y}-01-01", f"{y}-07-01"), (f"{y}-07-01", f"{y + 1}-01-01")):
            for q in WIN_QUERIES:
                try:
                    items += collect.gnews(f"{q} after:{h[0]} before:{h[1]}", "en", when=None, limit=40)
                except Exception as e:  # noqa: BLE001 - network hiccup on one window is fine
                    print("  skip", q, h, e)
    seen, uniq = set(), []
    for it in items:
        if it["title"].lower() not in seen:
            seen.add(it["title"].lower())
            uniq.append(it | {"id": len(uniq)})
    wins = []
    for i in range(0, len(uniq), 50):
        wins += [w for w in llm.harvest(uniq[i:i + 50]) if w.is_win]
    out = BT / "wins_candidates.csv"
    by_id = {u["id"]: u for u in uniq}
    best = {}
    for w in wins:  # earliest headline per customer+product
        k = (db.key(w.customer), w.product)
        if k not in best or w.date < best[k]["date"]:
            best[k] = {"customer": w.customer, "product": w.product, "date": by_id[w.id]["published"] or w.date,
                       "title": by_id[w.id]["title"], "url": by_id[w.id]["url"], "verified": ""}
    with out.open("w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=["customer", "product", "date", "title", "url", "verified"])
        cw.writeheader()
        cw.writerows(sorted(best.values(), key=lambda r: r["date"]))
    return out, len(uniq), len(best)


def backtest_press(con, limit=40):
    """For each past win: give the system only news from the 12 months before the announcement.
    Did it surface a buying signal for that customer, for the right product, and how early?"""
    path = BT / "wins.csv" if (BT / "wins.csv").exists() else BT / "wins_candidates.csv"
    wins = [w for w in csv.DictReader(path.open()) if w.get("verified", "").lower() != "no"][:limit]
    rows = []
    for w in wins:
        d = date.fromisoformat(w["date"])
        try:
            items = collect.gnews(f'"{w["customer"]}" after:{d - timedelta(days=365)} before:{d - timedelta(days=1)}',
                                  "en", when=None, limit=30)
        except Exception as e:  # noqa: BLE001
            print("  skip", w["customer"], e)
            continue
        items = [i | {"id": n} for n, i in enumerate(items) if i["published"] and i["published"] < w["date"]]
        tags = llm.tag(items) if items else []
        by_id = {i["id"]: i for i in items}
        mine = [t for t in tags if t.signal != "none" and t.confidence >= 0.5 and db.key(t.company) == db.key(w["customer"])]
        right = [t for t in mine if w["product"] in RULES.get(t.signal, {}) or t.product_hint == w["product"]]
        first = min((by_id[t.id]["published"] for t in mine), default=None)
        first_r = min((by_id[t.id]["published"] for t in right), default=None)
        rows.append({"customer": w["customer"], "product": w["product"], "date": w["date"], "items": len(items),
                     "any_signal": bool(mine), "right_product": bool(right),
                     "lead_months": round(months_between(date.fromisoformat(first), d), 1) if first else None,
                     "lead_months_right": round(months_between(date.fromisoformat(first_r), d), 1) if first_r else None})
        print(f"  {w['customer'][:30]:30} {w['date']} items={len(items):2} signal={bool(mine)} right={bool(right)}")
    leads = [r["lead_months"] for r in rows if r["lead_months"] is not None]
    out = {"file": path.name, "n": len(rows),
           "flagged_any": round(sum(r["any_signal"] for r in rows) / len(rows), 3) if rows else None,
           "flagged_right_product": round(sum(r["right_product"] for r in rows) / len(rows), 3) if rows else None,
           "median_lead_months": statistics.median(leads) if leads else None,
           "verified": path.name == "wins.csv", "rows": rows, "at": date.today().isoformat()}
    db.kv_set(con, "backtest_press", out)
    con.commit()
    return out
