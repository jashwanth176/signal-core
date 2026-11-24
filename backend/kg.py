"""Knowledge-graph utilities over data/graph.db: turn any name, id or misspelling into the right Singapore entity, and
answer what agents ask about it (who it is, who it is linked to, how two companies connect, what the news says).

Two derived structures, rebuilt by `build` after any registry reload:
  ids         crosswalk: (uen, scheme, value, source, at, conf) for LEIs, former names, hand aliases, GeBIZ suppliers
  entity_fts  trigram index over entity keys (contentless, rowid = entities.rowid), for misspellings
"""
import json
import re
from collections import Counter

from . import graph
from .db import ALIASES, key

# Shortcut: trading name -> registered name for the CRM accounts that differ; a pilot reads the UEN off the CRM record
REGISTERED = {"ninja van": "Ninja Logistics", "st engineering": "Singapore Technologies Engineering",
              "mapletree logistics trust": "Mapletree Logistics Trust Management", "sk gas": "SK Gas International",
              "tower transit": "Tower Transit Singapore", "becis energy as a service": "BECIS SEA",
              "sp": "Singapore Power", "sj": "Surbana Jurong Private",
              "singtel": "Singapore Telecommunications", "grab": "GrabTaxi Holdings",
              "pacific international lines": "Pacific International Lines (Private)"}


SCORE = {"uen": 1.0, "exact": 1.0, "alias": 1.0, "lei": 0.95, "former": 0.9, "typo": 0.8, "prefix": 0.5}
COLS = ("uen", "name", "key", "kind", "company_type", "status", "registered", "ssic", "ssic_desc", "activity", "postal",
        "officers", "former", "auditor")
# the old search order: live, the parent of the biggest group, a government supplier, holds an LEI (the operating
# company usually does, its shells rarely do), shortest name
RANK = ("NOT (e.status LIKE 'Live%'), -(SELECT count(*) FROM lei l JOIN owns o ON o.parent = l.lei "
        "WHERE l.reg_id = e.uen AND l.ra = 'RA000523'), NOT EXISTS(SELECT 1 FROM contracts WHERE supplier_key = e.key), "
        "NOT EXISTS(SELECT 1 FROM lei WHERE reg_id = e.uen AND ra = 'RA000523'), length(e.key), e.name")


def osa(a, b):
    """Optimal string alignment distance: insertions, deletions, substitutions, and a swap of neighbours, one each."""
    d = [[i + j if not i * j else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _near(a, b):
    return abs(len(a) - len(b)) <= 1 and osa(a, b) <= 1


def _typos(g, qk):
    """Entities one edit away from the query key. Candidates come from the 10 rarest of its trigrams (a misspelt
    trigram is rare or absent, the intact ones still find the name), best 200 by trigrams shared.
    Measured on the full graph (eval/resolve.py): 95% of one-letter typos right, 96% of foreign names declined, ~100 ms."""
    ts = sorted({qk[i:i + 3] for i in range(len(qk) - 2)})
    df = dict(g.execute(f"SELECT term, doc FROM entity_vocab WHERE term IN ({','.join('?' * len(ts))})", ts).fetchall())
    hits = Counter()
    for t in sorted((t for t in ts if df.get(t)), key=df.get)[:10]:
        hits.update(r[0] for r in g.execute("SELECT rowid FROM entity_fts WHERE entity_fts MATCH ?",
                                            ('"' + t.replace('"', '""') + '"',)))
    rowids = [i for i, _ in hits.most_common(200)]
    found = [u for u, k in g.execute(f"SELECT uen, key FROM entities WHERE rowid IN ({','.join('?' * len(rowids))})",
                                     rowids) if _near(qk, k)] if rowids else []
    return found + [u for u, v in g.execute("SELECT uen, value FROM ids WHERE scheme='alias'") if _near(qk, v)]


def resolve(g, q, k=5, prefix=False, typo=True, app=None):
    """Candidates for what a name or UEN means, best first: every entity column plus `score` and `via`
    (uen | exact | alias | lei | former | typo | prefix). Empty when nothing is close enough: the caller should say
    it does not know rather than guess. Typos are tried only when nothing matched outright; prefixes (for a search box
    as the user types) only when asked."""
    q = (q or "").strip()
    qk, qn, hits, curated = key(q), _norm(q), {}, []

    def add(rows, via):
        for (u,) in rows:
            if u and (u not in hits or hits[u][0] < SCORE[via]):
                hits[u] = (SCORE[via], via)

    if q:
        add(g.execute("SELECT uen FROM entities WHERE uen=? UNION SELECT reg_id FROM lei WHERE reg_id=? AND "
                      "ra='RA000523'", (q.upper(),) * 2), "uen")
    if qk:
        add(g.execute("SELECT uen FROM entities WHERE key=?", (qk,)), "exact")
        curated = [r[0] for r in g.execute("SELECT uen FROM ids WHERE scheme='alias' AND value IN (?, ?)",
                                            (qk, key(q, alias=False)))]  # 'SingPost' is stored as typed
        add([(u,) for u in curated], "alias")
        add(g.execute("SELECT reg_id FROM lei WHERE key=? AND ra='RA000523'", (qk,)), "lei")
        add(g.execute("SELECT uen FROM ids WHERE scheme='former' AND value=?", (qk,)), "former")
        if typo and not hits and len(qk) >= 3:
            add([(u,) for u in _typos(g, qk)], "typo")
        if prefix:
            add(g.execute(f"SELECT uen FROM entities e WHERE key BETWEEN ? AND ? ORDER BY {RANK} LIMIT ?",
                          (qk, qk + "~", k)), "prefix")
    rows = _rows(g, hits, qn, _words(q), set(curated))
    if app is not None:
        known={r['uen'] for r in rows}
        for saved in app.execute('SELECT uen,profile FROM knowledge_entities WHERE uen=? OR lower(name)=lower(?)'+(' OR name ILIKE ?' if prefix else '')+' ORDER BY name LIMIT ?',
                                 (q.upper(),q,q+'%',k) if prefix else (q.upper(),q,k)):
            if saved['uen'] not in known:
                e=json.loads(saved['profile'])['e']
                rows.append({c:e.get(c) for c in COLS}|{'score':1.0,'via':'collected registry'})
    return rows[:k]


def _words(s):
    w = " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))
    return w[4:] if w.startswith("the ") else w  # db.key drops a leading The too


def _said(r, qw, curated):
    """The name (or former name) starts with the words typed, or a curated alias says so. db.key drops Holdings,
    Corporation, Group..., so 'PSA International' and 'PSA Corporation' share a key: the typed words tell them apart."""
    return r["uen"] in curated or any(w == qw or w.startswith(qw + " ") for w in (_words(r["name"]), _words(r["former"])))


def _rows(g, hits, qn, qw="", curated=()):
    if not hits:
        return []
    marks = ",".join("?" * len(hits))
    rows = [dict(zip(COLS, r)) for r in g.execute(
        f"SELECT {', '.join('e.' + c for c in COLS)} FROM entities e WHERE uen IN ({marks}) ORDER BY {RANK}", list(hits))]
    got = {r["uen"] for r in rows}
    for uen, name, k, status, created in g.execute(  # in GLEIF but not the ACRA register: funds, VCCs
            f"SELECT reg_id, name, key, status, created FROM lei WHERE ra='RA000523' AND reg_id IN ({marks})", list(hits)):
        if uen not in got:
            got.add(uen)
            rows.append(dict.fromkeys(COLS) | {"uen": uen, "name": name, "key": k, "status": status, "registered": created})
    for i, r in enumerate(rows):
        r |= {"score": hits[r["uen"]][0], "via": hits[r["uen"]][1], "rank": i}
    # a name match (score >= 0.9) always beats a typo or a prefix; inside each, the words typed come first, then a
    # curated alias (a person said 'Grab' means GrabTaxi Holdings), then the full legal name typed
    rows.sort(key=lambda r: (r["score"] < 0.9, not (qw and _said(r, qw, curated)), r["uen"] not in curated, -r["score"],
                             _norm(r["name"]) != qn, r["rank"]))
    return [{c: v for c, v in r.items() if c != "rank"} for r in rows]


AS_OF = {"acra": "acra/%", "gleif": "%gleif%", "gebiz": "gebiz", "gdelt": "gdelt/%", "sgx": "sgx/%"}


def _record(g, uen):
    e = g.execute(f"SELECT {', '.join(COLS)} FROM entities WHERE uen=?", (uen,)).fetchone()
    if e:
        return dict(zip(COLS, e))
    lei = g.execute("SELECT reg_id, name, key, status, created FROM lei WHERE reg_id=? AND ra='RA000523'", (uen,)).fetchone()
    return dict.fromkeys(COLS) | dict(zip(("uen", "name", "key", "status", "registered"), lei)) if lei else None


def entity(g, uen, app=None):
    """Who a UEN is: registry fields, every name and id it goes by, and how fresh each source is."""
    e = _record(g, uen)
    if not e:
        if app is not None:
            from . import knowledge
            p = knowledge.base_profile(app, uen)
            if p:
                e = p['e']
                return {k:e.get(k) for k in ('uen','name','status','kind','registered','ssic','ssic_desc')} | {
                    'names':[{'name':e['name'],'via':'registry snapshot'}], 'ids':[], 'as_of':{'runtime':p.get('updated')}}
        return None
    ids = [dict(zip(("scheme", "value", "source", "conf"), r))
           for r in g.execute("SELECT scheme, value, source, conf FROM ids WHERE uen=? ORDER BY scheme, value", (uen,))]
    names = ([{"name": e["name"], "via": "registered"}] + ([{"name": e["former"], "via": "former"}] if e["former"] else [])
             + [{"name": i["value"], "via": "alias"} for i in ids if i["scheme"] == "alias"])
    return {c: e[c] for c in ("uen", "name", "status", "kind", "registered", "ssic", "ssic_desc")} | {
        "names": names, "ids": ids,
        "as_of": {s: g.execute("SELECT max(at) FROM loaded WHERE file LIKE ?", (f,)).fetchone()[0] for s, f in AS_OF.items()}}


def _keys(g, uen):
    e = _record(g, uen)
    if not e:
        return set()
    alias = {v for (v,) in g.execute("SELECT value FROM ids WHERE uen=? AND scheme='alias'", (uen,))}
    return {e["key"], key(e["former"] or ""), *alias} - {""}


def mentions(g, uen, since="", n=40, app=None):
    """News naming the company, newest first, each with why it matched: GDELT's own organisation tag, or the name in a
    Singapore headline (the same rule the company page uses). Wire copies of one story are kept once."""
    keys = _keys(g, uen)
    if not keys:
        if app is not None:
            from . import knowledge
            return [r|{'reason':'Persisted company monitor evidence'} for r in knowledge.overlay(app,uen)[0] if r['at']>=since][:n]
        return []
    rows, seen = [], set()
    for r in graph.news_for(g, sorted(keys), since, n * 2):
        if (t := (r["title"] or r["url"]).lower()) not in seen:
            seen.add(t)
            rows.append(r)
    rows = rows[:n]
    tags = dict(g.execute(f"SELECT news_id, org FROM mentions WHERE news_id IN ({','.join('?' * len(rows))}) "
                          f"AND org IN ({','.join('?' * len(keys))})", [r["id"] for r in rows] + sorted(keys)).fetchall()) if rows else {}
    words = sorted(keys | {a for a, c in ALIASES.items() if c in keys}, key=len, reverse=True)

    def why(r):
        if r["id"] in tags:
            return f"GDELT tagged it '{tags[r['id']]}'"
        w = next((w for w in words if w in (r["title"] or "").lower()), None)
        return f"headline names '{w}'" if w else "headline match"

    result = [{c: r[c] for c in ("id", "at", "url", "source", "lang", "title")} | {"reason": why(r)} for r in rows]
    if app is not None:
        from . import knowledge
        seen = {r['url'] for r in result}
        result += [r|{'reason':'Persisted company monitor evidence'} for r in knowledge.overlay(app,uen)[0] if r['url'] not in seen and r['at']>=since]
    return sorted(result,key=lambda r:r['at'] or '',reverse=True)[:n]


def filings(g, uen, since="", n=40):
    """The company's own SGX announcements, newest first: date, category, headline and the link to SGX's page."""
    keys = _keys(g, uen)
    return [{c: r[c] for c in ("id", "at", "issuer", "category", "title", "url")}
            for r in graph.filings_for(g, sorted(keys), since, n)] if keys else []


OWNERSHIP = (("parent", "child", "parent", "IS_DIRECTLY_CONSOLIDATED_BY"),
             ("ultimate_parent", "child", "parent", "IS_ULTIMATELY_CONSOLIDATED_BY"),
             ("subsidiary", "parent", "child", "IS_DIRECTLY_CONSOLIDATED_BY"))
# kinds a path may walk: ownership, people who moved, and every relation the app holds ("relations");
# GeBIZ agencies and news co-mentions link thousands of companies, so they would make every path two hops long
FLIP = {"parent": "subsidiary", "subsidiary": "parent", "acquirer": "acquired", "acquired": "acquirer",
        "supplier": "customer", "customer": "supplier"}
PATH_KINDS = ("parent", "ultimate_parent", "subsidiary", "alumni", "relations")


def _edge(src, dst, name, kind, source, at=None, conf=1.0, note=None):
    return {"src": src, "dst": dst, "name": name, "kind": kind, "source": source, "at": at, "conf": conf, "note": note}


def edges(g, app, node, kinds=None):
    """Everything one node is linked to. Nodes: a UEN, `lei:<LEI>` (a parent outside Singapore), `agency:<name>`,
    `company:<app id>` (an app company with no UEN), `org:<key>` (a name nothing resolves). Kinds, in this order:
    parent, ultimate_parent, subsidiary (GLEIF), awarded_by (GeBIZ, conf = how sure the supplier is this company),
    in_news_with (named together in 2+ articles), the app's own relations ("relations" selects all), alumni (people
    here who used to work there). `app` may be None: graph kinds only."""
    want = lambda k: kinds is None or k in kinds
    out, uen, cids = [], None, []
    if node.startswith("lei:"):
        leis = [node[4:]]
    elif node.startswith("company:"):
        leis, cids = [], [int(node[8:])] if app else []
    elif ":" in node:
        return []
    else:
        uen = node
        leis = [r[0] for r in g.execute("SELECT lei FROM lei WHERE reg_id=? AND ra='RA000523'", (uen,))]
        cids = [r[0] for r in app.execute("SELECT id FROM companies WHERE uen=?", (uen,))] if app else []
    for kind, me, other, rel in OWNERSHIP:
        if want(kind):
            for lei in leis:
                out += [_edge(node, reg if ra == "RA000523" else f"lei:{o}", name, kind, "gleif")
                        for o, name, ra, reg in g.execute(
                            f"SELECT l.lei, l.name, l.ra, l.reg_id FROM owns o JOIN lei l ON l.lei = o.{other} "
                            f"WHERE o.{me}=? AND o.kind=? AND o.status='PUBLISHED' ORDER BY l.name", (lei, rel))]
    if uen and want("awarded_by"):
        won = [(*r, conf) for v, conf in g.execute("SELECT value, conf FROM ids WHERE uen=? AND scheme='gebiz_supplier'", (uen,))
               for r in g.execute("SELECT agency, awarded, description, amount FROM contracts WHERE supplier_key=? AND supplier=?",
                                  (key(v), v))]
        out += [_edge(node, f"agency:{a}", a, "awarded_by", "gebiz", at, conf, f"{desc}, SGD {amt or 0:,.0f}")
                for a, at, desc, amt, conf in sorted(won, key=lambda w: w[1] or "", reverse=True)]
    if uen and want("in_news_with"):
        own, arts = _keys(g, uen), [a["id"] for a in mentions(g, uen, n=80)]
        seen = Counter(k for (orgs,) in g.execute(f"SELECT orgs FROM news WHERE id IN ({','.join('?' * len(arts))})", arts)
                       for k in {key(o) for o in (orgs or "").split(";")} if k and k not in own) if arts else Counter()
        for k, n in seen.most_common():
            if n >= 2:
                same = g.execute("SELECT uen, name FROM entities WHERE key=? LIMIT 2", (k,)).fetchall()
                dst, name = (same[0][0], same[0][1]) if len(same) == 1 else (f"org:{k}", k)
                out.append(_edge(node, dst, name, "in_news_with", "gdelt", conf=0.5, note=f"{n} articles"))
    for cid in cids:  # stored (src, dst, kind) reads "dst is src's kind"; from dst's side the kind flips
        rel = [(kind if src == cid else FLIP.get(kind, kind), source, oid, ouen, oname)
               for src, kind, source, oid, ouen, oname in app.execute(
                   "SELECT r.src, r.kind, r.source, c.id, c.uen, c.name FROM relations r JOIN companies c "
                   "ON c.id = CASE WHEN r.src=? THEN r.dst ELSE r.src END WHERE r.src=? OR r.dst=?", (cid,) * 3)]
        out += [_edge(node, ouen or f"company:{oid}", oname, kind, source) for kind, source, oid, ouen, oname in rel
                if kinds is None or "relations" in kinds or kind in kinds]
    if want("alumni"):
        for cid in cids:
            for person, hist, src in app.execute("SELECT name, history, source FROM people WHERE company_id=? "
                                                 "AND history IS NOT NULL", (cid,)):
                for h in json.loads(hist):
                    if h.get("current") or not h.get("organization_name"):
                        continue
                    r = resolve(g, h["organization_name"], 1, typo=False)
                    dst, name = (r[0]["uen"], r[0]["name"]) if r and r[0]["score"] >= 0.9 else (
                        f"org:{key(h['organization_name'])}", h["organization_name"])
                    if dst != uen:
                        out.append(_edge(node, dst, name, "alumni", "apollo" if src == "apollo" else "work history", h.get("end_date"),
                                         note=", ".join(filter(None, [person, h.get("title")]))
                                         + (f" until {h['end_date']}" if h.get("end_date") else "")))
    if app is not None and uen and want('evidence'):
        from . import knowledge
        out += [_edge(node,f"evidence:{d['id']}",d['title'],'evidence',d['url'],d['published'] or d['fetched'],note=d['text'][:500])
                for d in knowledge.documents(app,uen)]
    return out


def path(g, app, a, b, hops=3):
    """Shortest chain of links from a to b (ownership, people, app relations), as edges; [] when a is b, None when
    nothing joins them within `hops`."""
    if a == b:
        return []
    prev, frontier = {a: None}, [a]
    for _ in range(hops):
        nxt = []
        for node in frontier:
            es = edges(g, app, node, PATH_KINDS)
            if len(es) > 200 and node != a:  # Shortcut: hub cap; a group parent with 200+ links says little about a path
                continue
            for e in es:
                if e["dst"] in prev:
                    continue
                prev[e["dst"]] = e
                if e["dst"] == b:
                    chain, n = [], b
                    while prev[n]:
                        chain.append(prev[n])
                        n = prev[n]["src"]
                    return chain[::-1]
                nxt.append(e["dst"])
        frontier = nxt
    return None


def lookalikes(g, app, uens, k=20):
    """Live companies in the same industry (5-digit SSIC) as the seeds and not already in the app, most GeBIZ awards
    first, then holding a Singapore LEI, then oldest. Shortcut: industry code only; size and news fit come later."""
    seeds = [e for u in uens if (e := _record(g, u)) and e["ssic"]]
    if not seeds:
        return []
    by_code = {}
    for e in seeds:
        by_code.setdefault(e["ssic"], e)
    desc = {c: e["ssic_desc"] or (g.execute("SELECT ssic_desc FROM entities WHERE ssic=? AND ssic_desc IS NOT NULL "
                                            "LIMIT 1", (c,)).fetchone() or [None])[0] for c, e in by_code.items()}
    skip = {e["uen"] for e in seeds} | {u for (u,) in app.execute("SELECT uen FROM companies WHERE uen IS NOT NULL")}
    rows = g.execute(
        f"SELECT e.uen, e.name, e.ssic, (SELECT count(*) FROM ids i JOIN contracts c ON c.supplier_key = e.key "
        f"AND c.supplier = i.value WHERE i.uen = e.uen AND i.scheme = 'gebiz_supplier') AS awards, "
        f"EXISTS(SELECT 1 FROM lei WHERE reg_id = e.uen AND ra = 'RA000523') AS has_lei "
        f"FROM entities e WHERE e.ssic IN ({','.join('?' * len(by_code))}) AND e.status LIKE 'Live%' "
        f"AND e.uen NOT IN ({','.join('?' * len(skip))}) "
        f"ORDER BY awards DESC, has_lei DESC, e.registered IS NULL, e.registered LIMIT ?",
        (*by_code, *skip, k)).fetchall()
    out = []
    for uen, name, code, awards, has_lei in rows:
        seed, industry = by_code[code], desc[code] or "the same industry"
        why = f"Same industry as {seed['name']}: {industry.lower()} (SSIC {code})"
        if awards:
            why += f"; won {awards} government contract{'s' * (awards > 1)}"
        out.append({"uen": uen, "name": name, "ssic": code, "ssic_desc": desc[code], "awards": awards, "why": why,
                    "source": "ACRA, GeBIZ" if awards else "ACRA"})
    return out


def link(app, g):
    """Store each app company's UEN once, so pages, feeds and agents stop re-resolving names on every request.
    Only outright matches (score >= 0.9) are stored; a misspelt or unknown name stays unlinked."""
    n = 0
    for cid, name in app.execute("SELECT id, name FROM companies").fetchall():
        r = resolve(g, name, 1, typo=False)
        uen = r[0]["uen"] if r and r[0]["score"] >= 0.9 else None
        app.execute("UPDATE companies SET uen=COALESCE(?, uen) WHERE id=?", (uen, cid))  # keep a UEN the seller gave
        n += bool(uen)
    app.commit()
    return n


def _norm(name):
    return " ".join((name or "").upper().split())


def _best(g, k):
    """The entity a shared key most likely means: live, then holding a Singapore LEI, then oldest."""
    r = g.execute("SELECT uen FROM entities e WHERE key=? ORDER BY NOT (status LIKE 'Live%'), "
                  "NOT EXISTS(SELECT 1 FROM lei WHERE reg_id=e.uen AND ra='RA000523'), registered IS NULL, registered "
                  "LIMIT 1", (k,)).fetchone()
    return r[0] if r else None


def build(g):
    """Rebuild `ids` and `entity_fts` from what is loaded (~2 min on the full graph). An ACRA reload renumbers entity
    rowids, which the name index points at, so this runs after every registry load."""
    g.execute("DELETE FROM ids")
    g.execute("INSERT INTO ids SELECT reg_id, 'lei', lei, 'gleif', created, 1.0 FROM lei WHERE ra='RA000523'")
    g.executemany("INSERT INTO ids VALUES(?, 'former', ?, 'acra', ?, 1.0)",
                  [(u, fk, reg) for u, f, k, reg in g.execute(
                      "SELECT uen, former, key, registered FROM entities WHERE former IS NOT NULL").fetchall()
                   if (fk := key(f)) and fk != k])
    hand = [(a, t) for a, t in ALIASES.items()] + [(t, key(r)) for t, r in REGISTERED.items()]
    g.executemany("INSERT INTO ids VALUES(?, 'alias', ?, 'hand', NULL, 1.0)",
                  [(u, a) for a, t in hand if (u := _best(g, t))])
    rows = []
    for sup, sk, at in g.execute("SELECT supplier, supplier_key, max(awarded) FROM contracts "
                                 "WHERE supplier IS NOT NULL GROUP BY supplier").fetchall():
        cands = g.execute("SELECT uen, name FROM entities WHERE key=?", (sk,)).fetchall()
        exact = [u for u, n in cands if _norm(n) == _norm(sup)]
        if exact:
            rows.append((exact[0], sup, at, 1.0))
        elif cands:
            rows.append((_best(g, sk), sup, at, 1 / len(cands)))  # same name, several companies: the likeliest, flagged
    g.executemany("INSERT INTO ids VALUES(?, 'gebiz_supplier', ?, 'gebiz', ?, ?)", rows)
    # an SGX issuer's ISIN: the Singapore security it files under most (its shares, not one of its notes)
    isin = {}
    for k, i, at in g.execute("SELECT key, isin, max(at) FROM filings WHERE isin LIKE 'SG%' GROUP BY key, isin "
                              "ORDER BY count(*)").fetchall():
        isin[k] = (i, at)
    g.executemany("INSERT INTO ids VALUES(?, 'isin', ?, 'sgx', ?, 1.0)",
                  [(u, i, at[:10]) for k, (i, at) in isin.items() if (u := _best(g, k))])
    g.execute("INSERT INTO entity_fts(entity_fts) VALUES('delete-all')")
    g.execute("INSERT INTO entity_fts(rowid, key) SELECT rowid, key FROM entities WHERE length(key) >= 3")
    g.commit()
    n = dict(g.execute("SELECT scheme, count(*) FROM ids GROUP BY scheme").fetchall())
    return n | {"names": g.execute("SELECT count(*) FROM entities WHERE length(key) >= 3").fetchone()[0]}
