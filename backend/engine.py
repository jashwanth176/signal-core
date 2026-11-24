"""Signals -> opportunities -> action cards.

score = geometric mean of fit x intent x timing x relationship (each 0..1, all shown on the card).
intent = 1 - prod(1 - w(signal, product) * confidence * recency), with w learned from feedback (Beta update).
"""
import csv
import json
import math
import zlib
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from functools import lru_cache

from agents import gtm, llm
from . import db, model, people
from .catalog import (CONTACT_ROLE, DEFAULT_OWNER, EXCLUDE, GEBIZ_PRODUCT, HINT_PRIOR, HOLD_MONTHS, HOME, INDUSTRY_FIT,
                      PLACE_SIGNALS, PRODUCT_CATEGORY, PRODUCTS, RADAR_OPEN, REGION, RIVAL_BACK, RIVAL_TERM, RULES, SIGNALS,
                      SIZE_FIT, TERRITORY)
from .collect import GEBIZ, RENEWAL_SIM, TERM, add_months, is_singtel, similar

EVAL = db.ROOT / "eval"  # harvested past Singtel deals
NEAR = {"intl": "sdwan", "sdwan": "intl", "dc": "cloud", "cloud": "dc"}  # the closest product when a card's own has no published deal
K = 6  # prior strength: a rule weight counts as 6 pseudo-observations
BUMP = {"useful": (1, 0), "not_useful": (0, 1), "contacted": (0, 0), "meeting": (2, 0), "won": (3, 0), "lost": (0, 1)}
STAGES = ["contacted", "meeting", "won"]  # outcome funnel stages a seller marks on a card


def today():
    return date.today()


def months_between(a, b):
    return (b.year - a.year) * 12 + b.month - a.month + (b.day - a.day) / 30


# ---------------- tagging ----------------
def tag_items(con, limit=None):
    """Tag untagged items with Claude, 100 at a time, committing as we go."""
    done = 0
    while True:
        rows = con.execute("SELECT * FROM items WHERE tagged=0 ORDER BY id LIMIT 100").fetchall()
        if not rows or (limit and done >= limit):
            return done
        by_id = {r["id"]: r for r in rows}
        for t in llm.tag([dict(r) for r in rows]):
            it = by_id.get(t.id)
            if not it or t.signal == "none":
                continue
            cid = db.company(con, t.company, industry=t.industry, hq_country=t.hq_country.upper()[:2],
                             size=t.size) if t.company.strip() else None
            con.execute(
                "INSERT INTO signals(item_id,company_id,type,product_hint,summary,country,date,confidence,url,title,"
                "direction,part) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING",
                (it["id"], cid, t.signal, t.product_hint, t.summary_en, t.country.upper()[:2], it["published"],
                 min(max(t.confidence, 0), 1), it["url"], it["title"], t.direction, t.part.strip()))
            for rel in t.related:
                other = db.company(con, rel.name) if cid and rel.name.strip() else None
                if other and other != cid:
                    con.execute("INSERT INTO relations VALUES(?,?,?,?) ON CONFLICT DO NOTHING", (cid, other, rel.kind, f"news:{it['id']}"))
        con.executemany("UPDATE items SET tagged=1 WHERE id=?", [(i,) for i in by_id])
        con.commit()
        done += len(rows)


# ---------------- synthetic signals: CRM contract dates + GeBIZ renewal radar ----------------
def _sid(s):
    return -(zlib.crc32(s.encode()) & 0x7FFFFFFF)  # stable negative item_id for non-news signals


def radar_rows(con, lo=3, hi=RADAR_OPEN, ref=None):
    """Awards whose estimated end falls lo..hi months after ref and that have not been re-bought yet
    (no later, similar award at the same agency)."""
    ref = (ref or today()).isoformat()
    a0, a1 = add_months(date.fromisoformat(ref), lo).isoformat(), add_months(date.fromisoformat(ref), hi).isoformat()
    by_agency = defaultdict(list)
    for a in con.execute("SELECT * FROM awards WHERE award_date <= ? ORDER BY award_date", (ref,)):
        by_agency[a["agency"]].append(a)
    out = []
    for lst in by_agency.values():
        for i, a in enumerate(lst):
            if a0 <= a["est_end"] <= a1 and not any(
                    b["tender_no"] != a["tender_no"] and similar(a["description"], b["description"]) >= RENEWAL_SIM
                    for b in lst[i + 1:]):
                out.append(a)
    seen, uniq = set(), []
    for a in sorted(out, key=lambda a: (a["est_end"], -a["amount"])):  # one row per tender
        if a["tender_no"] not in seen:
            seen.add(a["tender_no"])
            uniq.append(a)
    return uniq


def upsert_signal(con, item_id, cid, typ, hint, summary, country, d, conf, url, title):
    """Insert or refresh a signal in place, keeping its id stable (cards cite ids)."""
    return con.execute(
        "INSERT INTO signals(item_id,company_id,type,product_hint,summary,country,date,confidence,url,title)"
        " VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(item_id,company_id,type) DO UPDATE SET product_hint=excluded.product_hint,"
        " summary=excluded.summary, date=excluded.date, confidence=excluded.confidence RETURNING id",
        (item_id, cid, typ, hint, summary, country, d, conf, url, title)).fetchone()[0]


def synth_signals(con):
    t = today()
    keep = []
    for c in con.execute("SELECT * FROM companies WHERE in_crm=1 AND contract_end IS NOT NULL").fetchall():
        end = date.fromisoformat(c["contract_end"])
        if not 0 <= months_between(t, end) <= RADAR_OPEN:
            continue
        ours = (c["incumbent"] or "").lower() == "singtel"
        summary = f"{'Our' if ours else c['incumbent'] or 'Current'} contract ends {end:%b %Y} (CRM)"
        keep.append(upsert_signal(con, _sid("crm|" + c["key"]), c["id"], "contract_ending", "none", summary,
                                  c["hq_country"], c["contract_end"], 0.9, None, "CRM contract record"))
    for a in radar_rows(con):
        cid = db.company(con, a["agency"], industry="government", hq_country="SG", size="large")
        holder = "Singtel group (defend)" if is_singtel(a["supplier"]) else a["supplier"].title()
        keep.append(upsert_signal(
            con, _sid(f"gebiz|{a['tender_no']}|{a['supplier']}"), cid, "contract_ending", GEBIZ_PRODUCT[a["category"]],
            f"{holder} holds {a['category']} contract '{a['description'][:90].strip()}' awarded {a['award_date']}, "
            + (f"~{a['term_months']} months" if TERM.search(a['description']) else f"term not stated, {a['term_months']} months assumed")
            + f"; estimated end {a['est_end']}", "SG", a["est_end"], 0.8,
            f"https://data.gov.sg/datasets/{GEBIZ}/view", f"GeBIZ {a['tender_no']}"))
    # a tech leader new in the job (from work history) reviews suppliers: the Unify-style "new hire" trigger
    for p in con.execute("SELECT p.*, c.hq_country FROM people p JOIN companies c ON c.id=p.company_id "
                         "WHERE p.started IS NOT NULL AND p.started <> ''").fetchall():
        if people.new_in_role(p["started"]) and people.TECH.search(p["title"] or ""):
            keep.append(upsert_signal(con, _sid("person|" + p["ext_id"]), p["company_id"], "leadership_change", "none",
                                      f"{p['name']} started as {p['title']} in {p['started']}", p["hq_country"],
                                      p["started"] + "-01", 0.8, p["linkedin"],
                                      "Work history" if p["source"] == "theorg" else "Apollo work history"))
    con.execute(f"DELETE FROM signals WHERE item_id < 0 AND id NOT IN ({','.join('?' * len(keep))})", keep)
    con.commit()


# ---------------- learned weights ----------------
def weights(con):
    w = {(s, p): [v * K, (1 - v) * K] for s, prods in RULES.items() for p, v in prods.items()}
    for s, v in HINT_PRIOR.items():
        for p in PRODUCTS:
            w.setdefault((s, p), [v * K, (1 - v) * K])
    for f in con.execute("SELECT f.kind, o.product, o.signal_ids FROM feedback f JOIN opps o ON o.id=f.opp_id"):
        a, b = BUMP[f["kind"]]
        for typ in {t for _, t in json.loads(f["signal_ids"])}:  # each signal type counts once per card
            ab = w.setdefault((typ, f["product"]), [0.3 * K, 0.7 * K])
            ab[0] += a
            ab[1] += b
    return {k: a / (a + b) for k, (a, b) in w.items()}


# ---------------- relationship graph ----------------
def graph(con):
    adj = defaultdict(list)
    for r in con.execute("SELECT src, dst, kind FROM relations"):
        adj[r["src"]].append((r["dst"], r["kind"]))
        adj[r["dst"]].append((r["src"], r["kind"]))
    singtel = con.execute("SELECT id FROM companies WHERE key='singtel'").fetchone()
    warm = {r["id"] for r in con.execute("SELECT id FROM companies WHERE is_customer=1")}
    if singtel:
        warm |= {d for d, k in adj[singtel["id"]]} | {singtel["id"]}
    return adj, warm


def warm_path(con, g, cid, depth=2):
    """Shortest path (<= depth hops) from a prospect to a Singtel customer, associate, or Singtel itself."""
    adj, warm = g
    prev, q = {cid: None}, deque([(cid, 0)])
    while q:
        node, d = q.popleft()
        if node in warm and node != cid:
            path = []
            while prev[node]:
                p, kind = prev[node]
                path.append((node, kind))
                node = p
            names = {r["id"]: r for r in con.execute(
                f"SELECT id, name, is_customer, owner FROM companies WHERE id IN ({','.join('?' * len(path))})",
                [n for n, _ in path])}
            return [{"name": names[n]["name"], "via": k, "customer": bool(names[n]["is_customer"]),
                     "owner": names[n]["owner"]} for n, k in reversed(path)]
        if d < depth:
            for nxt, kind in adj[node]:
                if nxt not in prev:
                    prev[nxt] = (node, kind)
                    q.append((nxt, d + 1))
    return None


# ---------------- scoring ----------------
def recency(s, t):
    if s["type"] == "contract_ending" or not s["date"]:
        return 1.0
    return math.exp(-max((t - date.fromisoformat(s["date"])).days, 0) / 60)


def fit(c):
    geo = 1.0 if c["hq_country"] == HOME else 0.85 if c["hq_country"] in REGION else 0.75 if not c["hq_country"] else 0.6
    return INDUSTRY_FIT.get(c["industry"], 0.5) * SIZE_FIT.get(c["size"], 0.7) * geo


def held(evidence, c, t):
    """One company's evidence split into what still argues for a call and the reasons not to call now:
    (evidence to score, {product category or "*" for every product: why to hold}, the cuts that removed evidence).
    A cut in one part of the company cancels growth evidence about that part dated up to the cut. A company-wide cut
    holds every card for HOLD_MONTHS. A contract just signed with another supplier holds that product's cards until
    RIVAL_BACK months before an assumed RIVAL_TERM-month term ends, then counts as a contract coming up for renewal.
    Evidence borrowed from a peer or a market rule is never a reason to hold.
    Shortcut: a part is the same when its name matches, or its country for growth tied to a place; a model of
    business lines if that misses."""
    part = lambda s: " ".join((s["part"] or "").lower().split())
    day = lambda s: s["date"][:10]
    own = lambda s, typ, way: s["company_id"] == c["id"] and typ == s["type"] and s["direction"] == way
    cuts = [s for s, typ, _ in evidence if own(s, typ, "shrink") and part(s)]
    keep, hold, used = [], {}, {}
    for s, typ, scale in evidence:
        if own(s, typ, "shrink"):
            if not part(s) and months_between(date.fromisoformat(day(s)), t) <= HOLD_MONTHS:
                hold["*"] = f"{s['summary']} ({day(s)})"
        elif own(s, typ, "rival"):
            end = add_months(date.fromisoformat(day(s)), RIVAL_TERM)
            back = add_months(end, -RIVAL_BACK)
            hint = s["product_hint"] if s["product_hint"] not in (None, "none") else None
            if t < back:
                hold[PRODUCT_CATEGORY[hint] if hint else "*"] = (
                    f"{s['summary']} ({day(s)}). Look again around {back:%b %Y}: a {RIVAL_TERM}-month term is assumed")
            else:
                keep.append((dict(s) | {"date": end.isoformat()}, "contract_ending", scale))
        else:
            cut = s["company_id"] == c["id"] and next((x for x in cuts if day(s) <= day(x) and (
                part(x) == part(s) or (typ in PLACE_SIGNALS and x["country"] and x["country"] == s["country"]
                                       and x["country"] != c["hq_country"]))), None)
            if cut:
                used[cut["id"]] = f"{cut['summary']} ({day(cut)})"
            else:
                keep.append((s, typ, scale))
    return keep, hold, list(used.values())


def timing(evidence, t, hold=None):
    if any(s["type"] == "tender_published" and recency(s, t) > 0.45 for s, _, _ in evidence):
        return 1.0, "Open tender"
    if hold:
        return 0.05, "Hold: " + hold
    # a news story tagged contract_ending carries the article's date, not the end date: only future ends count
    ends = [d for s, _, _ in evidence if s["type"] == "contract_ending" and (d := date.fromisoformat(s["date"])) >= t]
    if ends:
        m = months_between(t, min(ends))
        return max(0.35, math.exp(-((m - 9) / 6) ** 2)), f"Contract ends in ~{m:.0f} months"
    return 0.6, "No contract date known"


def relationship(c, path, gov_singtel):
    if c["is_customer"]:
        return 1.0, "Existing Singtel customer"
    if gov_singtel:
        return 0.85, "Singtel group already a supplier"
    if path:
        return 0.85, "Warm path via " + " > ".join(p["name"] for p in path)
    if c["in_crm"]:
        return 0.7, "Known account, no Singtel products yet"
    return 0.55, "Cold"


def gather(con, t):
    """company_id -> list of (signal_row, type_used, confidence_scale). Includes derived peer/market signals."""
    recent = add_months(t, -6).isoformat()
    ev = defaultdict(list)
    sigs = con.execute("SELECT s.*, c.industry FROM signals s LEFT JOIN companies c ON c.id=s.company_id "
                       "WHERE s.date >= ? OR s.type='contract_ending'", (recent,)).fetchall()
    for s in sigs:
        if s["company_id"]:
            ev[s["company_id"]].append((s, s["type"], 1.0))
    crm = con.execute("SELECT id, industry, hq_country FROM companies WHERE in_crm=1").fetchall()
    # a peer's breach in the last 60 days opens security budget at CRM accounts in the same industry
    for s in sigs:
        if s["type"] == "own_breach" and (s["date"] or "") >= add_months(t, -2).isoformat():
            for c in crm:
                if c["industry"] == s["industry"] and c["id"] != s["company_id"]:
                    ev[c["id"]].append((s, "peer_breach", 0.8))
    # a market-wide data rule applies to companies expanding into, or based in, that market
    for s in sigs:
        if s["type"] == "data_regulation" and not s["company_id"]:
            for cid, items in list(ev.items()):
                into = {x["country"] for x, typ, _ in items if typ in ("new_market_expansion", "new_facility")}
                if s["country"] in into:
                    ev[cid].append((s, "data_regulation", 0.9))
            for c in crm:
                if c["hq_country"] == s["country"]:
                    ev[c["id"]].append((s, "data_regulation", 0.9))
    return ev


def score(con):
    """Rebuild weights and opportunities from current signals + feedback."""
    t = today()
    synth_signals(con)
    W = weights(con)
    db.kv_set(con, "weights", {f"{s}|{p}": round(v, 3) for (s, p), v in W.items()})
    g = graph(con)
    wp = db.kv_get(con, "winprob")
    singtel_agencies = {r["agency"] for r in con.execute("SELECT agency, supplier FROM awards") if is_singtel(r["supplier"])}
    last_holder = {(a["agency"], a["category"]): a["supplier"] for a in con.execute("SELECT * FROM awards ORDER BY award_date")}
    run = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = 0
    for cid, evidence in gather(con, t).items():
        c = con.execute("SELECT * FROM companies WHERE id=?", (cid,)).fetchone()
        if c["key"] in EXCLUDE:
            continue
        evidence, hold, cuts = held(evidence, c, t)
        per = defaultdict(list)  # product -> [(signal, type, p)]
        for s, typ, scale in evidence:
            hint = s["product_hint"] if s["product_hint"] not in (None, "none") else None
            prods = [hint] if hint and typ in HINT_PRIOR else list(RULES.get(typ, {}))
            for p in prods:
                per[p].append((s, typ, W.get((typ, p), 0.3) * (s["confidence"] or 0.5) * scale * recency(s, t)))
        gov = c["industry"] == "government"
        path = None if c["is_customer"] else warm_path(con, g, cid)
        f = fit(c)
        for p, ev in per.items():
            # many articles about one event (same source company, type, country) count once, at their strongest.
            # Shortcut: no time window, so two real facility openings at one company merge; add a 90-day bucket if that bites
            best = {}
            for e in ev:
                k = (e[0]["company_id"], e[1], e[0]["country"])
                if k not in best or e[2] > best[k][2]:
                    best[k] = e
            ev = list(best.values())
            intent = 1 - math.prod(1 - x for _, _, x in ev)
            if intent < 0.2 or f < 0.3:
                continue
            cat = PRODUCT_CATEGORY[p]
            tm, tm_why = timing(ev, t, hold.get(cat) or hold.get("*"))
            holder = last_holder.get((c["name"], cat), "") if gov else (c["incumbent"] or "")
            incumbent = is_singtel(holder) or holder.lower() == "singtel"
            rel, rel_why = relationship(c, path, gov and c["name"] in singtel_agencies)
            sc = 100 * (f * intent * tm * rel) ** 0.25
            ev.sort(key=lambda x: -x[2])
            ids = json.dumps([[s["id"], typ] for s, typ, _ in ev])
            win = model.predict(wp, incumbent, bool(c["is_customer"]) or c["name"] in singtel_agencies, cat)
            meta = json.dumps({"timing_why": tm_why, "rel_why": rel_why, "warm_path": path,
                               "owner": c["owner"] or TERRITORY.get(c["industry"], DEFAULT_OWNER),
                               "defend": incumbent, "hold": tm_why.startswith("Hold: "), "cuts": cuts})
            con.execute(
                """INSERT INTO opps(company_id,product,score,fit,intent,timing,relationship,win_prob,signal_ids,meta,updated)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(company_id,product) DO UPDATE SET score=excluded.score,
                   fit=excluded.fit, intent=excluded.intent, timing=excluded.timing, relationship=excluded.relationship,
                   win_prob=excluded.win_prob, signal_ids=excluded.signal_ids, meta=excluded.meta, updated=excluded.updated""",
                (cid, p, round(sc, 1), round(f, 2), round(intent, 2), round(tm, 2), round(rel, 2), win, ids, meta, run))
            n += 1
    con.execute("DELETE FROM opps WHERE updated < ? AND id NOT IN (SELECT opp_id FROM feedback)", (run,))
    con.commit()
    return n


# ---------------- cards ----------------
def opp_view(con, o):
    """Everything the UI and the card writer need about one opportunity."""
    c = con.execute("SELECT * FROM companies WHERE id=?", (o["company_id"],)).fetchone()
    meta = json.loads(o["meta"] or "{}")
    pairs = json.loads(o["signal_ids"])
    rows = {r["id"]: r for r in con.execute(
        f"SELECT s.*, c.name AS company FROM signals s LEFT JOIN companies c ON c.id=s.company_id "
        f"WHERE s.id IN ({','.join('?' * len(pairs))})", [i for i, _ in pairs])}
    signals = [dict(rows[i]) | {"used_as": typ, "label": SIGNALS.get(typ, typ)} for i, typ in pairs if i in rows]
    cases = won_like(con, o["product"], c["industry"], c["id"])
    shown = published(o["product"])
    markets = db.markets(con)
    mk = sorted({s["country"] for s in signals if s["country"] in markets})
    return {"opp": dict(o), "company": dict(c), "meta": meta, "signals": signals, "cases": cases[:2], "published": shown,
            "markets": {m: markets[m] for m in mk}, "product_name": PRODUCTS[o["product"]],
            "role": CONTACT_ROLE[o["product"]], "card": json.loads(o["card"]) if o["card"] else None,
            "people": people.rank_for(people.at(con, c["id"]), o["product"])}


def won_like(con, product, industry, not_company, k=2):
    """Deals of this product marked won in Signal, same industry first: the only wins a card may cite."""
    return [{"title": f"{r['name']}: {PRODUCTS.get(product, product)}", "industry": r["industry"],
             "outcome": f"Marked won in Signal on {r['at'][:10]}"} for r in con.execute(
        "SELECT c.name, c.industry, max(f.at) AS at FROM feedback f JOIN opps o ON o.id=f.opp_id "
        "JOIN companies c ON c.id=o.company_id WHERE f.kind='won' AND o.product=? AND c.id<>? "
        "GROUP BY c.id, c.name, c.industry ORDER BY (c.industry=?) DESC, max(f.at) DESC LIMIT ?",
        (product, not_company, industry, k))]


@lru_cache
def _published():
    path = next((p for p in (EVAL / "wins.csv", EVAL / "wins_candidates.csv") if p.exists()), None)
    if not path:
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return sorted(csv.DictReader(f), key=lambda r: r["date"], reverse=True)


def published(product, k=1):
    """Singtel deals already announced in public for this product, else for the product nearest to it, newest first:
    what a seller can point a prospect to. From the harvested headlines in eval/ (wins.csv once someone has verified them).
    Shortcut: matched on the product only; an industry column in the file if cards should match on that too."""
    rows = _published()
    same = [r for r in rows if r["product"] == product] or [r for r in rows if r["product"] == NEAR.get(product)]
    return [{"title": r["title"], "industry": "", "outcome": f"Published {r['date']}", "url": r["url"]} for r in same[:k]]


def sig(o):
    """What a card was written from: its top evidence and the relationship reason."""
    return sorted(i for i, _ in json.loads(o["signal_ids"])[:3]) + [json.loads(o["meta"] or "{}").get("rel_why")]


def cards(con, top=30):
    """Write cards for the top opportunities whose evidence changed since the last card."""
    todo = []
    for o in con.execute("SELECT * FROM opps ORDER BY score DESC LIMIT ?", (top,)):
        if json.loads(o["meta"] or "{}").get("hold"):
            continue  # nothing to pitch while we hold
        old = json.loads(o["card"]) if o["card"] else {}
        if old.get("_sig") != sig(o):
            todo.append(opp_view(con, o))
    memo = gtm.context(con)  # read here: the writer threads below cannot use con

    def one(v):
        ctx = {
            "company": {k: v["company"][k] for k in ("name", "industry", "hq_country", "size", "is_customer",
                                                      "products_held", "incumbent", "contract_end", "notes")},
            "product": v["product_name"],
            "signals": [{k: s[k] for k in ("id", "label", "company", "date", "summary", "country")}
                        for s in v["signals"][:6]],
            "score_parts": {k: v["opp"][k] for k in ("fit", "intent", "timing", "relationship", "win_prob")},
            "timing": v["meta"].get("timing_why"), "relationship": v["meta"].get("rel_why"),
            "warm_path": v["meta"].get("warm_path"), "defend_existing_contract": v["meta"].get("defend"),
            "case_studies": v["cases"] + v["published"],
            "market_notes": {m: {"rules": p["rules"], "competitors": p["competitors"], "warm_path": p["warm_path"]}
                             for m, p in v["markets"].items()},
            "suggested_contact_role": v["role"],
            "from": v["meta"].get("owner"),
            "gtm_memory": memo,
        }
        return v["opp"]["id"], sig(v["opp"]), llm.write_card(ctx)

    with ThreadPoolExecutor(4) as ex:
        for oid, s, card in ex.map(one, todo):
            con.execute("UPDATE opps SET card=?, card_at=? WHERE id=?",
                        (json.dumps(card.model_dump() | {"_sig": s}), datetime.now(timezone.utc).isoformat(), oid))
            con.commit()
    return len(todo)


def report(con, cid):
    """Customer-first benchmark briefing built from peer signals."""
    c = con.execute("SELECT * FROM companies WHERE id=?", (cid,)).fetchone()
    peers = con.execute(
        "SELECT s.id, s.type, s.date, s.summary, s.country, c.name AS company FROM signals s JOIN companies c ON c.id=s.company_id "
        "WHERE c.industry=? AND c.id<>? AND s.type IN ('new_market_expansion','new_facility','own_breach','data_regulation',"
        "'digital_transformation','iot_rollout','merger_acquisition') ORDER BY s.date DESC LIMIT 40",
        (c["industry"], cid)).fetchall()
    rules = con.execute("SELECT id, type, date, summary, country FROM signals WHERE company_id IS NULL "
                        "ORDER BY date DESC LIMIT 10").fetchall()
    r = llm.write_report({"customer": {"name": c["name"], "industry": c["industry"], "hq": c["hq_country"]},
                          "peer_signals": [dict(x) for x in peers], "market_rules": [dict(x) for x in rules]})
    con.execute("INSERT INTO reports VALUES(?,?,?) ON CONFLICT(company_id) DO UPDATE SET body=excluded.body, at=excluded.at", (cid, r.model_dump_json(), date.today().isoformat()))
    con.commit()
    return r
