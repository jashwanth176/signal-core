"""GTM memory: what the team knows about selling Singtel Enterprise, in one shared place.

A section is its newest row in gtm_versions, so every edit keeps the old text. Seed text is copied from catalog.py
and seed/singtel-2026-11.md (web research the sections cite). This is text Claude reads; it does not
change scoring or play windows.
"""
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

from backend import db
from backend.catalog import COMMITTEE, CONTACT_ROLE, EXCLUDE, HOME, INDUSTRY_FIT, PRODUCTS, REGION, RULES, SIGNALS, SIZE_FIT

SCHEMA = """
CREATE TABLE IF NOT EXISTS gtm_versions(
  id BIGSERIAL PRIMARY KEY, section TEXT, title TEXT, body TEXT, source TEXT, at TEXT);
CREATE TABLE IF NOT EXISTS gtm_docs(id BIGSERIAL PRIMARY KEY, name TEXT, body TEXT, size BIGINT, at TEXT);
CREATE TABLE IF NOT EXISTS gtm_suggestions(
  id BIGSERIAL PRIMARY KEY, ref TEXT UNIQUE, section TEXT, headline TEXT, text TEXT, evidence TEXT,
  state TEXT DEFAULT 'waiting', edited BIGINT DEFAULT 0, at TEXT, decided TEXT);
"""
MAX_CUSTOM, MAX_BODY = 3, 20000
# the lookbehind keeps a long run of letters with no "@" linear (else each start position rescans it)
EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
RESEARCH = "seed/singtel-2026-11.md (web research, 6 Nov 2026)"


def _lines(*xs):
    return "\n".join(f"- {x}" for x in xs)


def _name(k):
    return k.replace("_", " ")


# (key, title, where it came from, seed text). Research lines are quoted as written there.
SECTIONS = [
    ("offer", "Singtel Enterprise offer", f"{RESEARCH}: What Singtel is, How it grows abroad, How enterprise sells, AI at Singtel",
     _lines("Enterprise is now over half of Singtel Singapore revenue; consumer is in a price war.",
            "Owns Singtel Singapore (consumer + enterprise, merged 2023), Optus (Australia), NCS (IT services), "
            "Digital InfraCo (Nxera data centres, RE:AI GPU cloud, Paragon).",
            "Own operations only in Singapore and Australia. Elsewhere: equity stakes, partner data centres "
            "(Johor with TM, Batam), GPU resale via Bridge Alliance.",
            "Government mostly through NCS via GeBIZ. NCS has 10 industry groups.",
            "RE:AI GPU cloud (6 MW fully contracted), Nvidia CoE (Feb 2026), Mistral and Scale AI partnerships.",
            'Investor Day 27 Aug 2026: "enterprise growth & AI adoption to mitigate consumer pressures"')),
    ("icp", "Ideal customer profile", "catalog.py: INDUSTRY_FIT, SIZE_FIT, HOME, REGION and EXCLUDE, the priors Signal scores with",
     _lines("Industries by fit, 1.0 is best: " + ", ".join(f"{_name(k)} {v}" for k, v in INDUSTRY_FIT.items()),
            "Company size by fit: " + ", ".join(f"{k} {v}" for k, v in SIZE_FIT.items()),
            f"Home market {HOME}. Region: " + ", ".join(sorted(REGION)),
            "Never pitched (the Singtel group, its associates and competitors): " + ", ".join(sorted(EXCLUDE)))),
    ("roles", "Buyer roles", f"committee.py role slots (catalog.COMMITTEE); {RESEARCH}: Champions and alumni",
     "Buying committee slots, filled from job titles in this order, one person per slot:\n"
     + _lines(*(label for _, label, _ in COMMITTEE),
              "Legal line: in government tenders use alumni for account context only, never to approach evaluators "
              "(Prevention of Corruption Act).")),
    ("signals", "Buying signals", "catalog.py SIGNALS, the signal types the news tagger uses",
     _lines(*(f"{_name(k).capitalize()}: {v}" for k, v in SIGNALS.items()))),
    ("market", "Singapore market", f"{RESEARCH}: Competitors, How enterprise sells, Champions and alumni",
     _lines("StarHub: enterprise order book up ~50% (1H2026), in talks to buy M1 (Sep 2026).",
            "Tata Communications: stronger on India-Singapore corridor and managed SD-WAN/SASE.",
            "Hyperscalers: AWS S$12bn into Singapore by 2028.",
            "Deal cycles 6-12 months, 12-18 for government.",
            "Gartner Jun 2025: 73% of B2B buyers avoid suppliers who send irrelevant outreach.",
            "NCS grew out of the National Computer Board, so ex-NCS people sit across GovTech and agencies.",
            "Business contact data is largely exempt from PDPA; profiling job histories from licensed data is fine, "
            "scraping LinkedIn is not.")),
    ("products", "Products", "catalog.py PRODUCTS, RULES (weight 0.5 or more) and CONTACT_ROLE",
     _lines(*(f"{name}: talk to {CONTACT_ROLE[p]}; strongest signals: " + ", ".join(
         _name(s) for s, w in sorted(((s, r[p]) for s, r in RULES.items() if r.get(p, 0) >= 0.5), key=lambda x: -x[1]))
         for p, name in PRODUCTS.items()))),
]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(text):
    """Saved text never keeps an email address (contact emails stay out of memory)."""
    return EMAIL.sub("[email removed]", (text or "").replace("\r\n", "\n")).strip()


def _insert(con, key, title, body, source):
    body = clean(body)
    if not body or len(body) > MAX_BODY:
        raise ValueError(f"text must be 1 to {MAX_BODY:,} characters")
    cur = con.execute("INSERT INTO gtm_versions(section, title, body, source, at) VALUES(?,?,?,?,?)",
                      (key, title, body, source, now()))
    con.commit()
    return cur.lastrowid


def seed(con):
    """First version of each built-in section that has none. Never overwrites an edit."""
    have = {r[0] for r in con.execute("SELECT DISTINCT section FROM gtm_versions")}
    todo = [s for s in SECTIONS if s[0] not in have]
    for key, title, source, body in todo:
        _insert(con, key, title, body, source)
    return len(todo)


def sections(con):
    """Built-in sections in order, then custom ones: newest text, its source and date, older versions newest first."""
    seed(con)
    by = {}
    for r in con.execute("SELECT * FROM gtm_versions ORDER BY id DESC"):
        by.setdefault(r["section"], []).append(r)
    order = [s[0] for s in SECTIONS] + sorted(k for k in by if k.startswith("custom-"))
    return [{"key": k, "title": by[k][0]["title"], "body": by[k][0]["body"], "source": by[k][0]["source"],
             "at": by[k][0]["at"], "custom": k.startswith("custom-"), "history": by[k][1:]} for k in order if k in by]


def save(con, key, body, title=None, source="Edited on this page"):
    """A new version of one section. Only custom sections can be renamed."""
    cur = con.execute("SELECT title FROM gtm_versions WHERE section=? ORDER BY id DESC", (key,)).fetchone()
    if not cur:
        raise ValueError(f"no section {key}")
    new_title = clean(title)[:80] if key.startswith("custom-") else ""
    return _insert(con, key, new_title or cur["title"], body, source)


def add_section(con, title, body):
    """One of up to 3 custom sections; returns its key."""
    title = clean(title)[:80]
    if not title:
        raise ValueError("a custom section needs a title")
    used = {r[0] for r in con.execute("SELECT DISTINCT section FROM gtm_versions WHERE section LIKE 'custom-%'")}
    key = next((k for k in (f"custom-{i}" for i in range(1, MAX_CUSTOM + 1)) if k not in used), None)
    if not key:
        raise ValueError(f"up to {MAX_CUSTOM} custom sections")
    _insert(con, key, title, body, "Added on this page")
    return key


# ---- documents: text, Markdown or CSV the team adds ----
DOC_TYPES, MAX_DOC = (".txt", ".md", ".csv"), 200_000


def docs(con):
    return con.execute("SELECT * FROM gtm_docs ORDER BY id DESC").fetchall()


def add_doc(con, name, body):
    name, body = clean(name).rsplit("/", 1)[-1][:120], clean(body)
    if not name.lower().endswith(DOC_TYPES):
        raise ValueError("a document must be .txt, .md or .csv")
    size = len(body.encode())
    if not body or size > MAX_DOC:
        raise ValueError(f"a document must be 1 to {MAX_DOC:,} bytes")
    cur = con.execute("INSERT INTO gtm_docs(name, body, size, at) VALUES(?,?,?,?)", (name, body, size, now()))
    con.commit()
    return cur.lastrowid


def remove_doc(con, did):
    con.execute("DELETE FROM gtm_docs WHERE id=?", (did,))
    con.commit()


# ---- suggestions: rule-based, from Signal's own outcomes; nothing changes until someone approves ----
# what sellers decide in the inbox, per origin (plays point at buying signals, card emails at products)
DECIDED = {"renewal_due": "Renewal packs", "new_leader": "Welcome notes", "peer_breach": "Briefing offers",
           "new_entity": "Intros to new entities", "abroad_entity": "Intros abroad", "card": "Card emails"}  # Shortcut: mirrors agents.PLAY_TITLE
MIN_DECIDED, DISMISS_SHARE, APPROVE_SHARE, MIN_WON = 4, 0.6, 0.8, 2  # Shortcut: hand-set thresholds, tune on real use


def _from_tasks(con):
    q = (f"SELECT t.origin, c.industry, count(*) AS n, count(*) FILTER (WHERE t.state='dismissed') AS d, count(*) FILTER (WHERE t.state='done') AS a, "
         f"max(t.decided) AS last FROM tasks t JOIN companies c ON c.id=t.company_id WHERE t.state IN ('done','dismissed') "
         f"AND c.industry NOT IN ('', 'other') AND t.origin IN ({','.join('?' * len(DECIDED))}) GROUP BY 1, 2 HAVING count(*) >= ?")
    for r in con.execute(q, (*DECIDED, MIN_DECIDED)):
        what, ind, n, last = DECIDED[r["origin"]], _name(r["industry"]), r["n"], r["last"][:10]
        if r["d"] / n >= DISMISS_SHARE:
            verb, k = "dismissed", r["d"]
        elif r["a"] / n >= APPROVE_SHARE:
            verb, k = "approved", r["a"]
        else:
            continue
        yield (f"tasks:{r['origin']}:{r['industry']}", "products" if r["origin"] == "card" else "signals",
               f"{what} were {verb} {k} of {n} times at {ind} accounts",
               f"- At {ind} accounts sellers {verb} {what.lower()} {k} of {n} times (to {last}).",
               f"Inbox decisions at {ind} accounts: {r['d']} dismissed, {r['a']} approved, last on {last}.")


def _from_backtest(con):
    b = db.kv_get(con, "backtest_gebiz")
    if not b or not b.get("flagged"):
        return
    span = f"{b['since']} to {b['data_end']}"
    lead = f", median {b['median_lead_months']} months before the award" if b.get("median_lead_months") is not None else ""
    yield (f"backtest:gebiz:{b['data_end']}", "signals",
           f"The renewal radar flagged {b['flagged']} of {b['renewal_wins']} Singtel renewal wins on GeBIZ before the award",
           f"- GeBIZ back-test ({span}): the renewal radar flagged {b['flagged']} of {b['renewal_wins']} renewal tenders "
           f"the Singtel group won{lead}.",
           f"GeBIZ back-test, {span}: {b['renewal_wins']} renewal tenders won, {b['flagged']} flagged ahead{lead}; "
           f"{b.get('defend', 0)} we already held, {b.get('attack', 0)} taken from another supplier.")


def _from_wins(con):
    won = defaultdict(list)
    for r in con.execute("SELECT c.industry, c.name, o.product, max(f.at) AS last FROM feedback f JOIN opps o ON o.id=f.opp_id "
                         "JOIN companies c ON c.id=o.company_id WHERE f.kind='won' AND c.industry NOT IN ('', 'other') "
                         "GROUP BY o.id, c.id"):
        won[r["industry"]].append(r)
    for industry, rows in won.items():
        if len(rows) < MIN_WON:
            continue
        ind, last = _name(industry), max(r["last"] for r in rows)[:10]
        made = ", ".join(f"{PRODUCTS.get(p, p)} {n}" for p, n in Counter(r["product"] for r in rows).most_common())
        yield (f"won:{industry}", "icp", f"{len(rows)} deals marked won at {ind} accounts",
               f"- Won at {ind} accounts: {len(rows)} deals ({made}), to {last}.",
               f"Marked won in Signal: {made}; accounts: {', '.join(sorted({r['name'] for r in rows}))}; last on {last}.")


def suggest(con):
    """Offer a card for every pattern found. A waiting card is refreshed in place (keeping a seller's edit); a decided
    topic never comes back, except the back-test, whose ref changes with each run."""
    n = 0
    for ref, section, headline, text, evidence in (*_from_tasks(con), *_from_backtest(con), *_from_wins(con)):
        n += con.execute(
            "INSERT INTO gtm_suggestions AS g(ref, section, headline, text, evidence, at) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(ref) DO UPDATE SET headline=excluded.headline, evidence=excluded.evidence, at=excluded.at, "
            "text=CASE WHEN g.edited=1 THEN g.text ELSE excluded.text END WHERE g.state='waiting' "
            "AND g.evidence != excluded.evidence",
            (ref, section, headline, text, evidence, now())).rowcount
    con.commit()
    return n


def suggestions(con, state="waiting"):
    return con.execute("SELECT * FROM gtm_suggestions WHERE state=? ORDER BY id DESC", (state,)).fetchall()


def decide(con, sid, action, text=None):
    """approve (appends the text, edited or not, to its section), edit (keeps it waiting) or dismiss."""
    s = con.execute("SELECT * FROM gtm_suggestions WHERE id=?", (sid,)).fetchone()
    if not s or action not in ("approve", "edit", "dismiss"):
        raise ValueError(f"cannot {action} suggestion {sid}")
    words = clean(text) or s["text"]
    if action == "approve":  # read (and seed) before the state flips: seeding commits
        body = next(x for x in sections(con) if x["key"] == s["section"])["body"] + "\n" + words
    if action == "edit":
        done = con.execute("UPDATE gtm_suggestions SET text=?, edited=1 WHERE id=? AND state='waiting'", (words, sid))
    else:  # the state flips first, so a second click finds nothing waiting
        done = con.execute("UPDATE gtm_suggestions SET state=?, text=?, decided=? WHERE id=? AND state='waiting'",
                           ("approved" if action == "approve" else "dismissed", words, now(), sid))
    if not done.rowcount:
        raise ValueError(f"suggestion {sid} was already decided")
    if action == "approve":
        try:
            save(con, s["section"], body, source=f"Suggestion approved: {s['headline']}")
        except Exception:  # autocommit: the flip already landed, so put it back
            con.execute("UPDATE gtm_suggestions SET state='waiting', decided=NULL WHERE id=?", (sid,))
            raise
    con.commit()
    return con.execute("SELECT * FROM gtm_suggestions WHERE id=?", (sid,)).fetchone()


# ---- readers ----
def context(con, limit=4000):
    """The memory as one compact text for Claude (card writer, draft writer, agent): sections in order, then documents.
    Short parts go in whole; long ones share what is left, so one big document cannot crowd out the sections."""
    parts = [f"## {s['title']}\n{s['body']}" for s in sections(con)]
    parts += [f"## Document: {d['name']}\n{d['body']}" for d in docs(con)]
    parts = [re.sub(r"\n\s*\n+", "\n", p) for p in parts]
    head = "GTM memory, kept by the Singtel Enterprise team (background on how we sell, not facts about any one account):\n"
    room, cut = limit - len(head) - 2 * len(parts), {}
    for i in sorted(range(len(parts)), key=lambda i: len(parts[i])):
        take = min(len(parts[i]), room // (len(parts) - len(cut)))
        cut[i] = parts[i] if take == len(parts[i]) else parts[i][:max(take - 1, 0)] + "…"
        room -= len(cut[i])
    return (head + "\n\n".join(cut[i] for i in range(len(parts)) if len(cut[i]) > 1))[:limit]
