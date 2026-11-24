"""Ask Signal: a seller states a goal, Claude works it with tools over the graph and the app DB, logs each step,
drops typed result cards and asks before anything spends credits or queues drafts in bulk.

Tools are plain functions `_tool_<name>(con, g, owner, tid, **args)`. `call` is the only way in: it never raises, and
every result is JSON the model can read, capped at MAX_RESULT characters. No tool returns a contact email."""
import json
import re
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

import anthropic

from backend import committee, db, filings, graph, kg, people
from . import accounts, llm

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads(id BIGSERIAL PRIMARY KEY, owner TEXT, title TEXT, state TEXT DEFAULT 'idle',
  messages TEXT, created TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS steps(id BIGSERIAL PRIMARY KEY, thread_id BIGINT, at TEXT, kind TEXT, text TEXT, body TEXT,
  ms BIGINT, state TEXT);
CREATE INDEX IF NOT EXISTS steps_thread ON steps(thread_id, id);
CREATE TABLE IF NOT EXISTS segments(id BIGSERIAL PRIMARY KEY, owner TEXT, name TEXT, items TEXT, step_id BIGINT UNIQUE,
  created TEXT);
"""
MAX_RESULT = 6000
MAX_ROUNDS = 8
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
SOURCE = {"gleif": "GLEIF", "gebiz": "GeBIZ", "gdelt": "GDELT news", "apollo": "Apollo", "mock_crm": "CRM",
          "public": "Public reports"}


class NotFound(Exception):
    """A company the registry or the seller's accounts do not have; the message goes back to the model as is."""


def _entity(g, uen, con=None):
    uen = (uen or '').strip().upper()
    e = kg._record(g, uen)
    if e or con is None:
        return e
    from backend import knowledge
    p = knowledge.base_profile(con,uen)
    return p['e'] if p else None


def _known(g, uen, con=None):
    e = _entity(g, uen, con)
    if not e:
        raise NotFound(f"No company with UEN {uen} in the registry. Use find_company to get the right UEN.")
    return e


def _account(con, g, uen):
    """The app account behind a UEN, or a plain refusal the model can pass on."""
    e = _known(g, uen, con)
    a = con.execute("SELECT * FROM companies WHERE uen=? ORDER BY in_crm DESC, id", (e["uen"],)).fetchone()
    if not a:
        raise NotFound(f"{e['name']} is not one of your accounts yet.")
    return a


def _tool_find_company(con, g, owner, tid, name):
    return {"results": [{"uen": r["uen"], "name": r["name"], "status": r["status"], "industry": r["ssic_desc"],
                         "matched_by": r["via"], "source": "ACRA" if r["kind"] else "GLEIF"}
                        for r in kg.resolve(g, name, 5, app=con)]}


def _tool_company_profile(con, g, owner, tid, uen):
    e = kg.entity(g, _known(g, uen, con)["uen"], app=con)
    a = con.execute("SELECT id, owner, is_customer, incumbent, products_held, contract_end FROM companies WHERE uen=? "
                    "ORDER BY in_crm DESC, id", (e["uen"],)).fetchone()
    return {"results": [e | {"source": "ACRA" if e["kind"] else "GLEIF",
                             "account": dict(a) | {"source": "CRM"} if a else None}]}


def _tool_company_links(con, g, owner, tid, uen, kinds=None):
    return {"results": [{"to": e["dst"], "name": e["name"], "kind": e["kind"], "date": e["at"], "conf": e["conf"],
                         "note": e["note"], "source": SOURCE.get(e["source"], e["source"])}
                        for e in kg.edges(g, con, _known(g, uen, con)["uen"], kinds)]}


def _tool_connection_path(con, g, owner, tid, a, b):
    chain = kg.path(g, con, _known(g, a)["uen"], _known(g, b)["uen"])
    if chain is None:
        return {"results": [], "note": "Nothing links them within 3 steps (ownership, people, CRM relations)."}
    return {"results": [{"from": e["src"], "to": e["dst"], "name": e["name"], "kind": e["kind"], "note": e["note"],
                         "source": SOURCE.get(e["source"], e["source"])} for e in chain]}


def _tool_company_news(con, g, owner, tid, uen, since=""):
    return {"results": [{"date": m["at"][:10], "title": m["title"], "site": m["source"], "why": m["reason"],
                         "source": m["url"]} for m in kg.mentions(g, _known(g, uen, con)["uen"], since, n=10, app=con)]}


def _tool_company_filings(con, g, owner, tid, uen, since=""):
    return {"results": [{"date": f["at"][:10], "category": f["category"], "title": f["title"], "filed_by": f["issuer"],
                         "source": f["url"]} for f in kg.filings(g, _known(g, uen)["uen"], since, n=15)]}


def _tool_read_filing(con, g, owner, tid, url):
    """Only documents Signal already holds a link to: the model never sends the box to a URL of its own making."""
    url = (url or "").strip()
    if not (g.execute("SELECT 1 FROM filings WHERE url=?", (url,)).fetchone()
            or con.execute("SELECT 1 FROM items WHERE url=? AND source IN ('nse','bse','sgx','asx')", (url,)).fetchone()):
        raise NotFound("Not a filing Signal holds. Use a source link from company_filings or account_signals.")
    try:
        text = filings.read(url, chars=MAX_RESULT - 1000)  # the whole answer must fit the cap
    except ImportError:  # the serverless copy has no document reader
        return {"results": [], "note": "Filings cannot be opened on this copy of Signal. Give the seller the link."}
    return {"results": [{"text": text, "source": url}] if text else [],
            "note": None if text else "The document has no readable text (a scan, or it could not be fetched)."}


def _tool_similar_companies(con, g, owner, tid, uens=(), k=10):
    if uens:
        seeds = [_known(g, u)["uen"] for u in uens]
    else:  # the seller's customers; a seller with none gets every Singtel customer as the seed
        q = "SELECT uen FROM companies WHERE is_customer=1 AND uen IS NOT NULL"
        seeds = [u for (u,) in con.execute(q + " AND owner=?", (owner,))] or [u for (u,) in con.execute(q)]
    return {"seeds": seeds, "results": kg.lookalikes(g, con, seeds, max(1, min(int(k), 25)))}


def _tool_account_people(con, g, owner, tid, uen):
    a = _account(con, g, uen)
    return {"results": [{"name": p["name"], "title": p["title"], "started": p["started"], "alumni": p["alumni"] or None,
                         "source": SOURCE.get(p["source"], "CRM")}
                        for p in con.execute("SELECT name, title, started, alumni, source FROM people WHERE company_id=? "
                                             "ORDER BY alumni DESC, started DESC", (a["id"],))]}


def _tool_account_signals(con, g, owner, tid, uen):
    a = _account(con, g, uen)
    return {"results": [{"type": s["type"], "summary": s["summary"], "date": s["date"], "source": s["url"]}
                        for s in con.execute("SELECT type, summary, date, url FROM signals WHERE company_id=? "
                                             "ORDER BY date DESC LIMIT 10", (a["id"],))]}


def _tool_my_inbox(con, g, owner, tid):
    return {"results": [{"company": t["company"], "title": t["title"], "why": t["why"], "source": "Signal inbox"}
                        for t in accounts.inbox(con, owner)["needs"][:20]]}


def _tool_account_plan(con, g, owner, tid, uen):
    a = _account(con, g, uen)
    out = []
    for s in accounts.plan(con, a["id"]):
        row = s["proposed"] or s["accepted"]
        if row:
            out.append({"section": s["heading"], "text": row["body"], "state": row["state"], "source": "Account plan"})
    return {"results": out}


# ---- result cards: rows the graph does not have and points with no source never reach the seller ----
def _card(con, g, tid, name, args, body):
    return {"shown": len(body.get("items") or body.get("points") or [1]),
            "card": step(con, tid, "card", say(con, g, name, args), body)}


def _tool_show_accounts(con, g, owner, tid, title, accounts):
    items, seen = [], set()
    for a in accounts:
        e = _entity(g, a.get("uen"))
        if e and a.get("source") and e["uen"] not in seen:
            seen.add(e["uen"])
            acct = con.execute("SELECT id FROM companies WHERE uen=? ORDER BY in_crm DESC, id", (e["uen"],)).fetchone()
            items.append({"uen": e["uen"], "name": e["name"], "why": a.get("why") or "", "source": a["source"],
                          "account_id": acct["id"] if acct else None})
    if not items:
        return {"shown": 0, "note": "None of those companies is in the registry with a source, so no list was shown."}
    ids = [x["account_id"] for x in items if x["account_id"]]
    intent = {r["company_id"]: r["i"] for r in con.execute(
        f"SELECT company_id, max(intent) i FROM opps WHERE company_id IN ({','.join('?' * len(ids))}) GROUP BY company_id",
        ids)} if ids else {}
    for x in items:  # intent is Signal's own score for the account, never a number the model wrote
        x["intent"] = round(intent[x["account_id"]] * 100) if intent.get(x["account_id"]) else None
    return _card(con, g, tid, "show_accounts", {"title": title}, {"type": "accounts", "title": title, "items": items})


def _tool_show_brief(con, g, owner, tid, uen, title, points):
    e = _known(g, uen)
    keep = [{"text": p["text"], "source": p["source"]} for p in points if p.get("text") and p.get("source")]
    if not keep:
        return {"shown": 0, "note": "Every point needs a source; none had one, so no brief was shown."}
    return _card(con, g, tid, "show_brief", {"uen": e["uen"]},
                 {"type": "brief", "uen": e["uen"], "name": e["name"], "title": title, "points": keep})


def _tool_build_workflow(con, g, owner, tid, spec):
    flows = db.feature("flows")
    if not flows:
        return {"error": "Workflows are not set up here."}
    wid = flows.from_spec(con, owner, spec)
    w = flows.get(con, wid)
    out = _card(con, g, tid, "build_workflow", {"name": w["name"]},
                {"type": "workflow", "workflow_id": wid, "name": w["name"], "steps": len(spec.get("steps") or []),
                 "trigger": (spec.get("trigger") or {}).get("type", "manual")})
    flows.ask_publish(con, wid)
    sid = step(con, tid, "approval", "Publish this workflow?", {
        "action": "publish", "workflow_id": wid, "gets": f"{w['name']} starts running on its trigger",
        "cost": " ".join(flows.promises(flows.graph_of(w), w["budget"]))}, state="waiting")
    return out | {"waiting": sid, "note": WAIT}


def _tool_show_draft(con, g, owner, tid, uen, to_role, subject, body, sources=()):
    e = _known(g, uen)
    a = con.execute("SELECT id FROM companies WHERE uen=? ORDER BY in_crm DESC, id", (e["uen"],)).fetchone()
    return _card(con, g, tid, "show_draft", {"uen": e["uen"], "to_role": to_role},
                 {"type": "draft", "uen": e["uen"], "name": e["name"], "account_id": a["id"] if a else None,
                  "to_role": to_role, "subject": subject, "body": body,
                  "sources": [s for s in sources if isinstance(s, str) and s.strip()]})


# ---- approvals: anything that spends credits or fills the inbox waits for the seller ----
WAIT = "Nothing happens until the seller approves."


def _tool_request_people_lookup(con, g, owner, tid, uen, max_people=3):
    a = _account(con, g, uen)
    if not a["domain"]:
        raise NotFound(f"{a['name']} has no website on file, so there is nobody to look up yet.")
    n = max(1, min(int(max_people), 5))
    sid = step(con, tid, "approval", say(con, g, "request_people_lookup", {"uen": uen}),
               {"action": "people", "company_id": a["id"], "name": a["name"], "k": n, "cost": f"Up to {n} Apollo credits",
                "gets": f"Up to {n} senior contacts at {a['name']}, with titles"}, state="waiting")
    return {"waiting": sid, "note": WAIT}


def _tool_request_queue_drafts(con, g, owner, tid, drafts):
    keep = {}
    for d in drafts:
        a = _account(con, g, d["uen"])
        if not (d.get("subject") and d.get("body")):
            raise ValueError("every draft needs a subject and a body")
        if a["id"] in keep:
            raise ValueError(f"two drafts for {a['name']}: one draft per account")
        keep[a["id"]] = {"company_id": a["id"], "uen": a["uen"], "name": a["name"], "subject": d["subject"],
                         "body": d["body"]}
    if not keep or len(keep) > 20:
        raise ValueError("between 1 and 20 drafts, one per account")
    n = f"{len(keep)} draft{'s' * (len(keep) != 1)}"
    sid = step(con, tid, "approval", say(con, g, "request_queue_drafts", {"n": n}),
               {"action": "queue", "cost": "No credits", "gets": f"{n} in your inbox, one per account",
                "drafts": list(keep.values())}, state="waiting")
    return {"waiting": sid, "note": WAIT}


# ---- questions, account searches and goals the model can put to the seller ----
def _tool_ask_seller(con, g, owner, tid, question, options):
    opts = [str(o).strip()[:80] for o in options if str(o).strip()]
    if not 2 <= len(opts) <= 4:
        raise ValueError("between 2 and 4 options")
    sid = step(con, tid, "ask", question.strip()[:200], {"model": True, "options": [{"label": o, "value": o} for o in opts]},
               state=WAITING)
    return {"waiting": sid, "note": "The question is on screen. End your turn now: the seller's answer comes back as "
                                    "their next message."}


def _tool_find_accounts(con, g, owner, tid, source, months=6):
    if source not in SEARCH:
        raise ValueError(f"source is one of {', '.join(SEARCH)}")
    ctx = {"months": max(1, min(int(months), 24)), "ranked": {r["company_id"]: r for r in _ranked(con, owner)}}
    rows, said, _ = SEARCH[source](con, g, owner, ctx)
    return {"said": said, "results": rows}


def _tool_renewal_pack(con, g, owner, tid, limit=10, months=18):
    from backend import renewals
    pack = renewals.pack(con, owner, limit, months)
    if tid:
        sid = step(con, tid, "card", "Renewal brief", {"type": "renewals", **pack})
        return {"card": sid, "total": pack["total"], "note": "The sourced renewal cards are on screen. Use delegate_research if the seller requests recurring background research and delivery to Home."}
    return pack


def _tool_schedule_automation(con, g, owner, tid, time="09:00", until="this_week", limit=10, months=18, salesforce=False):
    from . import auto
    return auto.renewal_request(con, owner, tid, time, until, limit, months, salesforce)


def _tool_delegate_research(con, g, owner, tid, instructions, time='09:00', until='this_week', limit=10, criteria=None, uens=None, include_customers=False):
    from . import revenue
    return revenue.propose(con,owner,tid,instructions,time,until,limit,criteria,uens,include_customers)


def _tool_set_goal(con, g, owner, tid, metric, target, deadline):
    flows = db.feature("flows")
    if not flows:
        return {"error": "Goals are not set up here."}
    if metric not in flows.METRICS:
        raise ValueError(f"metric is one of {', '.join(flows.METRICS)}")
    date.fromisoformat(deadline)
    gid = flows.add_goal(con, owner, metric, int(target), deadline, None)
    label = f"{int(target)} {flows.METRICS[metric]} by {_fmt(deadline)}"
    sid = step(con, tid, "result", f"Goal set: {label}. Signal checks in every week.",
               {"feature": "goal", "objects": [_obj("goal", "Live", f"/workflows?as={quote(owner)}&tab=goals")],
                "rows": None, "goal": gid})
    return {"card": sid, "goal": gid}


def _p(**props):
    return {"type": "object", "properties": {k: v if isinstance(v, dict) else {"type": "string", "description": v}
                                             for k, v in props.items()},
            "required": [k for k, v in props.items() if not (isinstance(v, dict) and v.pop("optional", False))]}


UEN = "Singapore UEN, exactly as find_company returned it"
TOOLS = [
    {'name':'delegate_research','description':'Delegate real background prospecting, monitoring, historical research, renewal/competitor intelligence, verified people/introductions and editable sales emails. Use for daily new-company or autonomous research requests. Work continues with the browser closed. Shows approval and queues durable work only; never claim research complete or emails sent. until is inclusive YYYY-MM-DD or this_week. Reuse sourced contacts and discover new people through permitted free search. Paid reveal requires a separate bounded seller grant. No automatic email or Salesforce sending. End after this approval.',
     'input_schema':{'type':'object','required':['instructions','time','until'],'properties':{
         'instructions':{'type':'string'},'time':{'type':'string','description':'HH:MM Singapore time'},'until':{'type':'string'},'limit':{'type':'integer'},
         'criteria':{'type':'object','properties':{'words':{'type':'array','items':{'type':'string'}},'registered_after':{'type':'string'},'gov_supplier':{'type':'boolean'},'contract_words':{'type':'array','items':{'type':'string'}},'parent_country':{'type':'string'}}},
         'uens':{'type':'array','items':{'type':'string'},'description':'Only explicit registry UENs the seller supplied; otherwise use graph candidates.'},'include_customers':{'type':'boolean','description':'True only if seller requests existing customers too.'}}}},
    {"name": "renewal_pack", "description": "Top renewal contracts from the seller's CRM buyers and GeBIZ radar, soonest first; includes incumbent, estimated dates, signals, people, warm introduction, sourced research, pitch and next step. Shows cards. No paid lookups.",
     "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}, "months": {"type": "integer"}}}},
    {"name": "find_company", "description": "Find Singapore companies by name, trading name, former name or UEN, with "
     "typo tolerance. Returns up to 5 candidates, best first; empty when nothing is close enough.",
     "input_schema": _p(name="Company name or UEN as the seller typed it")},
    {"name": "company_profile", "description": "Registry record for one company (status, industry, names, ids, how fresh "
     "each source is) plus the seller's CRM account if it has one (owner, customer or not, products, contract end).",
     "input_schema": _p(uen=UEN)},
    {"name": "company_links", "description": "Everything one company is linked to: GLEIF parents and subsidiaries, GeBIZ "
     "government contracts (awarded_by), companies named with it in the news, CRM relations, people who used to work at "
     "other companies. Big companies are cut short (truncated).",
     "input_schema": _p(uen=UEN, kinds={"type": "array", "items": {"type": "string", "enum": [
         "parent", "ultimate_parent", "subsidiary", "awarded_by", "in_news_with", "relations", "alumni"]},
         "description": "Only these kinds of link", "optional": True})},
    {"name": "connection_path", "description": "Shortest chain of links (ownership, people, CRM relations) from company a "
     "to company b, up to 3 steps.", "input_schema": _p(a=UEN, b=UEN)},
    {"name": "company_news", "description": "Up to 10 news articles naming the company, newest first, with why each matched.",
     "input_schema": _p(uen=UEN, since={"type": "string", "description": "Only news on or after this date, YYYY-MM-DD",
                                            "optional": True})},
    {"name": "company_filings", "description": "Up to 15 of the company's own SGX announcements, newest first: date, "
     "category, headline and the link to SGX. Only SGX-listed companies have any.",
     "input_schema": _p(uen=UEN, since={"type": "string", "description": "Only filings on or after this date, YYYY-MM-DD",
                                            "optional": True})},
    {"name": "read_filing", "description": "The text of one stock exchange filing (the first pages), to quote the "
     "detail behind a headline: deal size, country, who was appointed. Takes a source link from company_filings, or "
     "from an account signal whose source is an NSE, BSE, SGX or ASX filing.",
     "input_schema": _p(url="The filing's source link, exactly as given")},
    {"name": "similar_companies", "description": "Live Singapore companies in the same industry as the seed companies and "
     "not yet in Signal, government suppliers first. With no seeds, uses the seller's own customers.",
     "input_schema": _p(uens={"type": "array", "items": {"type": "string"}, "description": "Seed UENs",
                                  "optional": True},
                        k={"type": "integer", "description": "How many, 1-25 (default 10)", "optional": True})},
    {"name": "account_people", "description": "People on file at one of the seller's accounts: name, title, when they "
     "started, past employers that matter.", "input_schema": _p(uen=UEN)},
    {"name": "account_signals", "description": "The 10 newest buying signals recorded for one of the seller's accounts.",
     "input_schema": _p(uen=UEN)},
    {"name": "my_inbox", "description": "Tasks waiting on the seller in their Signal inbox.", "input_schema": _p()},
    {"name": "account_plan", "description": "The account plan for one of the seller's accounts, section by section.",
     "input_schema": _p(uen=UEN)},
    {"name": "show_accounts", "description": "Show the seller a list of companies as a card. Every row needs the UEN, why it "
     "is on the list in a few words, and the source of that reason (e.g. GeBIZ, ACRA, a news URL). Rows the registry "
     "does not have, or with no source, are dropped.",
     "input_schema": _p(title="Short title for the list", accounts={"type": "array", "description": "The rows", "items": {
         "type": "object", "required": ["uen", "why", "source"], "properties": {
             "uen": {"type": "string"}, "why": {"type": "string"}, "source": {"type": "string"}}}})},
    {"name": "show_brief", "description": "Show the seller a short brief on one company as a card: up to 6 points, each "
     "with its source. Points with no source are dropped.",
     "input_schema": _p(uen=UEN, title="Short title", points={"type": "array", "description": "The points", "items": {
         "type": "object", "required": ["text", "source"], "properties": {
             "text": {"type": "string"}, "source": {"type": "string"}}}})},
    {"name": "show_draft", "description": "Show the seller an email draft for one company as a card. Under 120 words, "
     "addressed to a role, never an email address. The seller can queue it to their inbox from the card.",
     "input_schema": _p(uen=UEN, to_role="The job title it is for", subject="Subject line", body="The email",
                        sources={"type": "array", "items": {"type": "string"}, "description": "Where its facts come from",
                                 "optional": True})},
    {"name": "request_people_lookup", "description": "Ask the seller to approve finding senior contacts at one of their "
     "accounts through Apollo (costs credits). Nothing happens until they approve. End your turn after it: the result "
     "comes back as the seller's next message.",
     "input_schema": _p(uen=UEN, max_people={"type": "integer", "description": "1-5 (default 3)", "optional": True})},
    {"name": "build_workflow", "description": "Save a workflow the seller described as a draft on their Workflows page. "
     "It never runs until the seller approves the publish card this posts; end your turn after it. spec = {name, budget (max Claude calls per run, "
     "default 20), trigger: {type: schedule, every: day|week, day: 0-6 Monday=0, hour: 0-23 Singapore} | {type: signal, "
     "signal: a buying-signal type} | {type: segment, segment_id} | {type: task, decision: approved|dismissed|any} | "
     "{type: manual}, steps: [{type, id?, after? (step id, or 'trigger'), branch?, ...settings}]}. Step types and "
     "settings: condition rules=[{field, op, value}] (branches yes/no); ask question, options=[2-4] (one branch per "
     "option); graph what=entity|links|news (found/none); people product (approved/skipped); draft about; queue title; "
     "enrol sequence; notify channel=slack|teams, text ('{account}' is replaced); delay days, hours.",
     "input_schema": _p(spec={"type": "object", "description": "The workflow, shaped as above"})},
    {"name": "request_queue_drafts", "description": "Ask the seller to approve putting several email drafts in their "
     "inbox, one per account (only accounts they already have). Nothing is ever sent: approved drafts wait in the "
     "inbox for the seller's Send. End your turn after it.",
     "input_schema": _p(drafts={"type": "array", "description": "Up to 20 drafts", "items": {
         "type": "object", "required": ["uen", "subject", "body"], "properties": {
             "uen": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}}})},
    {"name": "ask_seller", "description": "Put a choice to the seller as buttons: which product, how many accounts, "
     "which role, whether to set a goal. Then end your turn; their answer is their next message.",
     "input_schema": _p(question="One short question", options={"type": "array", "items": {"type": "string"},
                                                                  "description": "2-4 short options"})},
    {"name": "find_accounts", "description": "The seller's own accounts by what is happening there, each row with why "
     "and its source, and intent 0-100 when an opportunity scores it. source: signals (a buying signal on file, "
     "strongest first), renewals (a contract ending in the next `months`), committee (no senior tech, security or "
     "infrastructure contact on file), changed (new signals this week), inbox (work waiting), lookalikes (companies "
     "like their customers, not in Signal yet).",
     "input_schema": _p(source={"type": "string", "enum": ["signals", "renewals", "committee", "changed", "inbox",
                                                           "lookalikes"]},
                        months={"type": "integer", "description": "renewals only: months ahead (default 6)",
                                "optional": True})},
    {"name": "set_goal", "description": "Set a goal the seller agreed to, checked every week. metric: replies (Signal "
     "cannot see meetings, so a meetings goal counts replies), emails_approved, emails_sent, senior_contacts, "
     "plays_completed or intros.",
     "input_schema": _p(metric="One of the metrics above", target={"type": "integer", "description": "How many"},
                        deadline="YYYY-MM-DD")},
]
def _cap(out):
    """Cut the results list so the JSON fits MAX_RESULT; say so, and how many there were."""
    if len(json.dumps(out, default=str)) <= MAX_RESULT or not isinstance(out.get("results"), list):
        return out
    rows, keep, size = out["results"], [], len(json.dumps({k: v for k, v in out.items() if k != "results"}, default=str)) + 80
    for r in rows:
        size += len(json.dumps(r, default=str)) + 2
        if size > MAX_RESULT:
            break
        keep.append(r)
    return out | {"results": keep, "truncated": True, "total": len(rows)}


def call(con, g, owner, tid, name, args):
    """Run one tool for the model. Never raises: an unknown tool, bad arguments or an unknown company come back as
    {"error": ...} so the model can correct itself."""
    fn = globals().get(f"_tool_{name}")
    if not fn:
        return {"error": f"No tool called {name}."}
    try:
        return _cap(fn(con, g, owner, tid, **(args or {})))
    except NotFound as e:
        return {"error": str(e)}
    except (TypeError, ValueError, KeyError) as e:
        return {"error": f"Bad input for {name}: {e}"}
    except Exception as e:  # a failing lookup must not end the seller's turn; the model hears it and moves on
        traceback.print_exc()
        return {"error": f"{name} failed: {e.__class__.__name__}"}


# the step log line for each tool; card and approval tools write it on their own card instead of a work step
SAY = {"find_company": 'Looked up "{name}" in the registry', "company_profile": "Read the registry record for {uen}",
       "company_links": "Listed what {uen} is linked to", "connection_path": "Looked for a path from {a} to {b}",
       "company_news": "Read the news about {uen}", "company_filings": "Read the SGX filings of {uen}",
       "read_filing": "Opened a filing", "similar_companies": "Looked for companies like {uens}",
       "account_people": "Checked who we know at {uen}", "account_signals": "Read the buying signals for {uen}",
       "my_inbox": "Read your inbox", "account_plan": "Read the account plan for {uen}",
       "show_accounts": "{title}", "show_brief": "A brief on {uen}", "show_draft": "A draft for {to_role} at {uen}",
       "build_workflow": "A workflow draft: {name}", "request_people_lookup": "Find senior contacts at {uen}?", "request_queue_drafts": "Put {n} in your inbox?",
       "find_accounts": "Searched your accounts: {what}"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _name(con, g, uen):
    a = con.execute("SELECT name FROM companies WHERE uen=? ORDER BY in_crm DESC, id", (uen,)).fetchone()
    e = None if a else _entity(g, uen)
    return a["name"] if a else e["name"] if e else uen


def say(con, g, name, args):
    a = dict(args or {})
    for k in ("uen", "a", "b"):
        if isinstance(a.get(k), str):
            a[k] = _name(con, g, a[k].strip().upper())
    a["what"] = LIST_NAME.get(a.get("source"), "accounts").lower()
    a["uens"] = ", ".join(_name(con, g, str(u).strip().upper()) for u in a.get("uens") or []) or "your customers"
    try:
        return SAY[name].format(**a)
    except (KeyError, IndexError, ValueError):
        return f"Used {name.replace('_', ' ')}"


def new(con, owner, text):
    t = now()
    cur = con.execute("INSERT INTO threads(owner, title, state, messages, created, updated) VALUES(?,?,'idle','[]',?,?)",
                      (owner, text.strip()[:80], t, t))
    con.commit()
    return cur.lastrowid


def step(con, tid, kind, text, body=None, ms=None, state=None):
    """One line of the thread's log. Email addresses never reach a step, whoever wrote them."""
    hide = lambda s: EMAIL.sub("[email hidden]", s) if s else s
    t = now()
    cur = con.execute("INSERT INTO steps(thread_id, at, kind, text, body, ms, state) VALUES(?,?,?,?,?,?,?)",
                      (tid, t, kind, hide(text), hide(json.dumps(body, default=str)) if body is not None else None, ms,
                       state))
    con.execute("UPDATE threads SET updated=? WHERE id=?", (t, tid))
    con.commit()
    return cur.lastrowid


def _save(con, tid, msgs):
    con.execute("UPDATE threads SET messages=?, updated=? WHERE id=?", (json.dumps(msgs), now(), tid))
    con.commit()


def _content(text, extra):
    """The seller's message as the model reads it: files first, then one context line per tagged object, the scope,
    and their words. The bubble on screen shows only their words, with the tags and files as chips."""
    lines = [r["context"] for r in extra.get("refs") or [] if r.get("context")]
    if extra.get("scope") == "mine":
        lines.append("Scope: only the seller's own accounts.")
    said = "\n".join(["Attached by the seller:", *lines, "", text]) if lines else text
    return [{"type": "file", **f} for f in extra.get("files") or []] + [{"type": "text", "text": said}]


def run(tid, text, extra=None):
    """One turn: the seller's message, then up to MAX_ROUNDS model rounds of tool calls, then the answer. Opens its own
    connections (it runs in a worker thread). Messages are saved only where every tool_use has its result. extra comes
    from claim(): the user step is then already written."""
    if not llm.live():
        raise RuntimeError("no key")
    con, g = db.connect(), graph.connect()
    if not con.execute("UPDATE threads SET state='running', updated=? WHERE id=? AND state IN ('idle', 'queued')",
                       (now(), tid)).rowcount:
        con.commit()
        raise RuntimeError("busy")
    t = con.execute("SELECT * FROM threads WHERE id=?", (tid,)).fetchone()
    msgs = json.loads(t["messages"] or "[]") + [{"role": "user", "content": _content(text, extra) if extra else text}]
    _save(con, tid, msgs)
    memo = (gtm := db.feature("gtm")) and gtm.context(con)
    from .auto import SGT
    system = llm.AGENT_SYSTEM + f"\n\nToday in Singapore is {datetime.now(SGT).date().isoformat()}. The seller is {t['owner']}; sign drafts with that name. " + (
        "For delegated background research, monitoring, prospecting or daily new-company work use delegate_research. "
        "This performs real collection after approval; do not replace it with a saved renewal lookup or a workflow draft. "
        "Use until=this_week for this week. Companies must be new to this mission, never recycle yesterday's list. "
        "No paid people lookups, automatic email or Salesforce writes are included. End after its approval card.") + (f"\n\nWhat the team knows about selling (GTM memory, approved by sellers):\n{memo}" if memo else "")
    if extra is None:
        step(con, tid, "user", text)
    con.execute("UPDATE steps SET state='skipped' WHERE thread_id=? AND kind='ask' AND state=?", (tid, WAITING))
    con.commit()  # an open write here locks every other connection (the model call counter) out
    try:
        for _ in range(MAX_ROUNDS):
            r = llm.agent_turn(system, msgs, TOOLS)
            if r["stop_reason"] == "refusal":
                step(con, tid, "stopped", "The model declined this one. Try asking it another way.")
                break
            blocks = r["content"]
            msgs.append({"role": "assistant", "content": blocks})
            words = "\n\n".join(b["text"].strip() for b in blocks if b["type"] == "text" and b["text"].strip())
            uses = [b for b in blocks if b["type"] == "tool_use"]
            if not uses:
                _save(con, tid, msgs)
                if words:
                    step(con, tid, "answer", words)
                break
            if words:
                step(con, tid, "note", words)
            results, renewal_approval = [], False
            for b in uses:
                t0 = time.monotonic()
                out = call(con, g, t["owner"], tid, b["name"], b["input"])
                renewal_approval |= b["name"] in ('schedule_automation','delegate_research') and bool(out.get("waiting"))
                if not (out.get("card") or out.get("waiting") or out.get("quiet")):
                    found = out.get("error") or out.get("note") or f"{len(out.get('results', []))} found"
                    step(con, tid, "work", say(con, g, b["name"], b["input"]), {"tool": b["name"], "found": found},
                         round((time.monotonic() - t0) * 1000))
                results.append({"type": "tool_result", "tool_use_id": b["id"], "content": json.dumps(out, default=str)}
                               | ({"is_error": True} if "error" in out else {}))
            msgs.append({"role": "user", "content": results})
            _save(con, tid, msgs)
            if renewal_approval:
                break
        else:
            step(con, tid, "stopped", f"Stopped after {MAX_ROUNDS} steps. Ask again to carry on from here.")
    except anthropic.APIError as e:
        print(f"thread {tid}: model call failed: {e!r}")
        step(con, tid, "stopped", "The model did not answer this time. Ask again in a moment.")
    except Exception:
        traceback.print_exc()
        step(con, tid, "stopped", "Something went wrong on our side. Ask again.")
    finally:
        con.execute("UPDATE threads SET state='idle', updated=? WHERE id=?", (now(), tid))
        con.commit()


WAITING = "waiting"
LIST_NAME = {"signals": "High-intent accounts", "lookalikes": "Companies like your customers", "renewals": "Renewals",
             "committee": "Missing a senior IT contact", "changed": "New signals", "inbox": "Waiting in your inbox"}
def _today():
    return date.today().isoformat()


def _fmt(d):
    return date.fromisoformat(str(d)[:10]).strftime("%-d %b %Y")


def _obj(feature, label, href=None):
    """An object card on a result: the feature it opens, a short count or state, and where Open goes."""
    return {"feature": feature, "label": label, "href": href}


def _icp(con):
    """Best-fit industries from the GTM memory's ICP section ("logistics 1.0, manufacturing 1.0, ...")."""
    gtm = db.feature("gtm")
    text = next((s["body"] for s in gtm.sections(con) if s["key"] == "icp"), "") if gtm else ""
    return [k.strip() for k in re.findall(r"([a-z][a-z ]*?) 1\.0\b", text.lower())]


# -- searches: (rows, what was found in one sentence, where Open goes). A row is one company: company_id (None when
#    Signal does not hold it), uen, name, why and its source, and intent 0-100 when an opportunity scores it.
def _ranked(con, owner):
    """The seller's accounts with a scored opportunity, strongest intent first, each with the signal behind it."""
    best = {}
    for o in con.execute("SELECT o.*, c.name, c.uen, c.industry FROM opps o JOIN companies c ON c.id=o.company_id "
                         "WHERE o.intent > 0 ORDER BY o.intent DESC, o.score DESC"):
        if o["company_id"] not in best and json.loads(o["meta"] or "{}").get("owner") == owner:
            best[o["company_id"]] = o
    cites = {cid: ids[0][0] for cid, o in best.items() if (ids := json.loads(o["signal_ids"] or "[]"))}
    rest = [cid for cid in best if cid not in cites]
    cited = {r["id"]: r for r in con.execute(
        f"SELECT id, summary, url FROM signals WHERE id IN ({','.join('?' * len(cites))})", list(cites.values()))}
    latest = {r["company_id"]: r for r in con.execute(
        f"SELECT DISTINCT ON (company_id) company_id, summary, url FROM signals WHERE company_id IN "
        f"({','.join('?' * len(rest))}) AND date <= ? ORDER BY company_id, date DESC", (*rest, _today()))}
    rows = []
    for cid, o in best.items():
        s = cited.get(cites[cid]) if cid in cites else latest.get(cid)
        rows.append({"company_id": cid, "uen": o["uen"], "name": o["name"], "industry": o["industry"],
                     "intent": round(o["intent"] * 100), "product": o["product"],
                     "why": s["summary"] if s else f"Scored {round(o['intent'] * 100)} on intent for {o['product']}",
                     "source": (s and s["url"]) or "Signal score"})
    return rows


def _mine(con, owner):
    return [cid for cid, o in accounts.scope(con).items() if o == owner]


def _companies(con, cids):
    cids = list(cids)
    return {r["id"]: r for r in con.execute(f"SELECT id, name, uen FROM companies WHERE id IN ({','.join('?' * len(cids))})",
                                            cids)}


def _row(c, why, source, ranked, **more):
    """A search result row for company row c (id, name, uen)."""
    r = ranked.get(c["id"]) or {}
    return {"company_id": c["id"], "uen": c["uen"], "name": c["name"], "why": why, "source": source,
            "intent": r.get("intent"), "product": r.get("product")} | more


def _s_signals(con, g, owner, ctx):
    rows = [r for r in _ranked(con, owner) if not ctx.get("industry") or r["industry"] == ctx["industry"]]
    what = f"{ctx['industry'].replace('_', ' ')} accounts" if ctx.get("industry") else "accounts"
    return rows, (f"Found {len(rows)} of your {what} with a buying signal on file, scored by intent." if rows else
                  f"None of your {what} has a buying signal on file yet."), "/plays"


def _s_lookalikes(con, g, owner, ctx):
    seed = (ctx.get("company") or {}).get("uen")
    q = "SELECT uen, name FROM companies WHERE is_customer=1 AND uen IS NOT NULL"
    seeds = ([(seed, ctx["company"]["name"])] if seed else
             con.execute(q + " AND owner=?", (owner,)).fetchall() or con.execute(q).fetchall())
    rows = [{"company_id": None, "uen": r["uen"], "name": r["name"].title(), "why": r["why"], "source": r["source"],
             "intent": None, "product": None} for r in kg.lookalikes(g, con, [u for u, _ in seeds], 25)]
    like = " and ".join(n for _, n in seeds[:2]) + (" and others" if len(seeds) > 2 else "")
    return rows, (f"Found {len(rows)} live companies in the same industries as {like} that Signal does not hold yet."
                  if rows else f"No live company in the same industries as {like or 'your customers'} turned up "
                               "outside Signal."), None


def _s_renewals(con, g, owner, ctx):
    m, today = int(ctx["months"]), _today()
    end, ranked, rows = (date.today() + timedelta(days=round(30.4 * m))).isoformat(), ctx.get("ranked") or {}, []
    mine, hits = _mine(con, owner), {}
    marks = ",".join("?" * len(mine))
    for c in con.execute(f"SELECT id, contract_end FROM companies WHERE id IN ({marks}) "
                         "AND substr(contract_end, 1, 10) BETWEEN ? AND ?", (*mine, today, end)):
        hits.setdefault(c["id"], []).append((c["contract_end"][:10], f"Our contract ends {_fmt(c['contract_end'])}", "CRM"))
    for s in con.execute(f"SELECT company_id, date, summary, url FROM signals WHERE company_id IN ({marks}) "
                         "AND type='contract_ending' AND date BETWEEN ? AND ?", (*mine, today, end)):
        hits.setdefault(s["company_id"], []).append((s["date"], s["summary"], s["url"] or "Signal"))
    names = _companies(con, hits)
    for cid in (c for c in mine if c in hits):
        d, why, src = min(hits[cid])
        rows.append(_row(names[cid], why, src, ranked, date=d))
    rows.sort(key=lambda r: r["date"])
    return rows, (f"Found {len(rows)} of your accounts with a contract ending in the next {m} months, soonest first."
                  if rows else f"No contract at your accounts ends in the next {m} months."), "/radar"


def _s_committee(con, g, owner, ctx):
    mine, ranked = _mine(con, owner), ctx.get("ranked") or {}
    gaps = {cid: [s["label"].lower() for s in slots if s["slot"] in ("tech", "security", "infra") and not s["person"]]
            for cid, slots in committee.slots_for(con, mine).items()}
    names = _companies(con, [cid for cid in mine if gaps[cid]])
    rows = [_row(names[cid], "No " + ", ".join(gaps[cid]) + " contact on file", "Signal contacts", ranked,
                 gaps=len(gaps[cid])) for cid in mine if gaps[cid]]
    rows.sort(key=lambda r: (-r["gaps"], -(r["intent"] or 0)))
    return rows, (f"{len(rows)} of your {len(mine)} accounts are missing a tech, security or infrastructure contact."
                  if rows else "Every one of your accounts has a senior IT contact on file."), "/team"


def _s_changed(con, g, owner, ctx):
    mine, ranked = set(_mine(con, owner)), ctx.get("ranked") or {}
    for days in (7, 30):
        sigs = [s for s in con.execute("SELECT * FROM signals WHERE date BETWEEN ? AND ? ORDER BY date DESC",
                                       (accounts.ago(days), _today())) if s["company_id"] in mine]
        if sigs:
            break
    first = {}
    for s in sigs:
        first.setdefault(s["company_id"], s)
    names = _companies(con, first)
    rows = {cid: _row(names[cid], s["summary"], s["url"] or "Signal", ranked) for cid, s in first.items()}
    return list(rows.values()), (f"{len(sigs)} new signal{'s' * (len(sigs) != 1)} at {len(rows)} of your accounts in "
                                 f"the last {days} days." if rows else
                                 "No new signals at your accounts in the last 30 days."), f"/brief/{quote(owner)}"


def _s_inbox(con, g, owner, ctx):
    ranked, first = ctx.get("ranked") or {}, {}
    for t in accounts.inbox(con, owner)["needs"]:
        first.setdefault(t["company_id"], t)
    names = _companies(con, first)
    rows = {cid: _row(names[cid], t["title"], "Signal inbox", ranked) for cid, t in first.items()}
    rows = sorted(rows.values(), key=lambda r: -(r["intent"] or 0))
    return rows, (f"{len(rows)} of your accounts have work waiting in your inbox." if rows else
                  "Nothing is waiting in your inbox."), f"/brief/{quote(owner)}#needs"


SEARCH = {"signals": _s_signals, "lookalikes": _s_lookalikes, "renewals": _s_renewals, "committee": _s_committee,
          "changed": _s_changed, "inbox": _s_inbox}


def pick(con, sid, value):
    """The seller's answer to a question card the model put. The first answer counts; the option goes back to the
    model as the seller's next message."""
    s = con.execute("SELECT * FROM steps WHERE id=? AND kind='ask'", (sid,)).fetchone()
    b = json.loads(s["body"]) if s else {}
    opt = next((o for o in b.get("options") or [] if o["value"] == value), None)
    if not opt or not con.execute("UPDATE steps SET state='answered' WHERE id=? AND state=?", (sid, WAITING)).rowcount:
        con.commit()
        raise ValueError(f"step {sid} is not waiting on that answer")
    _set_body(con, sid, b | {"picked": opt["label"]})
    return opt


def suggestions(con, owner):
    """Starter prompts for this seller, each counted from their accounts and inbox (the best-fit industries come from
    the GTM memory's ICP). They only fill the composer; the model works whichever the seller sends."""
    ranked, out = _ranked(con, owner), []
    by = {r["company_id"]: r for r in ranked}
    if ranked:
        out.append({"intent": "meetings", "text": f"Get meetings with my {len(ranked)} accounts showing buying signals"})
        for ind in _icp(con):
            n = sum((r["industry"] or "").replace("_", " ") == ind for r in ranked)
            if 0 < n < len(ranked):
                out.append({"intent": "meetings", "text": f"Get meetings with my {n} {ind} accounts showing buying signals"})
                break
    names = [n for (n,) in con.execute("SELECT name FROM companies WHERE is_customer=1 AND owner=? AND uen IS NOT NULL "
                                       "ORDER BY name", (owner,))]
    out.append({"intent": "lookalikes", "text": "Find companies like my best customers"
                + (f", {' and '.join(names[:2])}" if names else "")})
    for m in (6, 12):
        rows = _s_renewals(con, None, owner, {"months": m, "ranked": by})[0]
        if rows:
            out.append({"intent": "renewals", "text": f"Line up the {len(rows)} renewal{'s' * (len(rows) != 1)} in the "
                                                      f"next {m} months"})
            break
    gaps = _s_committee(con, None, owner, {"ranked": by})[0]
    if gaps:
        out.append({"intent": "committee", "text": f"Find senior IT buyers at my {len(gaps)} account"
                                                   f"{'s' * (len(gaps) != 1)} missing one"})
    top = next((t["company"] for t in accounts.inbox(con, owner)["needs"]), None) or (ranked[0]["name"] if ranked else None)
    if top:
        out.append({"intent": "brief", "text": f"Brief me on {top}"})
    return out[:6]


def goal_view(con, steps):
    """Goal cards render live: each goal a result step set, measured now (counts from the inbox, never invented)."""
    flows, out = db.feature("flows"), {}
    for s in steps:
        gid = s["body"] and s["kind"] == "result" and json.loads(s["body"]).get("goal")
        gl = gid and flows and con.execute("SELECT * FROM goals WHERE id=?", (gid,)).fetchone()
        if gl:
            v = flows.measure(con, gl)
            out[gid] = {"value": v, "target": gl["target"], "label": flows.METRICS[gl["metric"]], "state": gl["state"],
                        "deadline": _fmt(gl["deadline"]), "pct": min(100, round(100 * v / max(gl["target"], 1))),
                        "weekly": "every week"}
    return out


STALE = timedelta(minutes=5)


def recover(con, tid):
    """A restart mid-turn leaves a thread running with nobody working it. Five quiet minutes: say so and free it."""
    t = con.execute("SELECT state, updated FROM threads WHERE id=?", (tid,)).fetchone()
    if not t or t["state"] not in ("running", "queued"):
        return False
    last = max(filter(None, (t["updated"], con.execute("SELECT max(at) FROM steps WHERE thread_id=?", (tid,)).fetchone()[0])))
    if datetime.fromisoformat(last) > datetime.now(timezone.utc) - STALE:
        return False
    if not con.execute("UPDATE threads SET state='idle' WHERE id=? AND state IN ('running', 'queued') AND updated=?",
                       (tid, t["updated"])).rowcount:
        con.commit()
        return False
    step(con, tid, "stopped", "This stopped before it finished. Ask again to carry on.")
    return True


def start(tid, text, extra=None, force=False):
    """Queue the turn: the seller's message shows at once as a pending user step, and the page's stream claims it
    (claim) and runs it while it streams the steps. extra holds tagged objects, files and scope (see _content). Raises
    RuntimeError('busy') if the thread is already working. Past Signal's daily model cap the turn waits on an approval
    card instead (force: the seller approved it)."""
    con = db.connect()
    flows = db.feature("flows")
    if flows and not force and (n := flows.calls_today(con)) >= flows.CAP:
        step(con, tid, "approval", f'Today\'s model budget is used up. Answer "{text.strip()[:80]}" anyway?', {
            "action": "budget", "text": text, "extra": extra, "cost": f"Up to {MAX_ROUNDS} model calls, past the daily cap of {flows.CAP} "
                                                      f"({n} made today)", "gets": "The agent carries on with this message"},
             state=WAITING)
        return
    ok = con.execute("UPDATE threads SET state='queued', updated=? WHERE id=? AND state='idle'", (now(), tid)).rowcount
    con.commit()
    if not ok:
        raise RuntimeError("busy")
    step(con, tid, "user", text, extra or None, state="pending")


def claim(con, tid):
    """The turn start() queued, for the one stream that wins it: (text, extra), or None when there is none."""
    s = con.execute("SELECT id, text, body FROM steps WHERE thread_id=? AND kind='user' AND state='pending' "
                    "ORDER BY id DESC", (tid,)).fetchone()
    won = s and con.execute("UPDATE steps SET state=NULL WHERE id=? AND state='pending'", (s["id"],)).rowcount
    con.commit()
    return (s["text"], json.loads(s["body"] or "{}")) if won else None


def _thread_of(con, sid):
    return con.execute("SELECT t.* FROM threads t JOIN steps s ON s.thread_id = t.id WHERE s.id=?", (sid,)).fetchone()


def _set_body(con, sid, body, state=None):
    con.execute("UPDATE steps SET body=?, state=coalesce(?, state) WHERE id=?", (json.dumps(body, default=str), state, sid))
    con.commit()


def _queue(con, cid, owner, t, sid, d):
    why = f"Drafted in Ask Signal: {t['title']}"
    return accounts.queue(con, cid, owner, "agent", "email", f"thread:{t['id']}:{sid}", f"Email ready: {d['name']}", why,
                        [{"label": "Ask Signal", "url": f"/ask/{t['id']}"}],
                        {"subject": d["subject"], "draft": d["body"], "to": d.get("to_role"),
                         "to_person_id": d.get("to_person_id")})


def decide(con, sid, action):
    """The seller's answer to an approval card. Only the request that flips it out of waiting acts, so a double click
    runs the lookup or the queueing once; the second gets ValueError."""
    if action not in ("approve", "skip"):
        raise ValueError(f"cannot {action}")
    proposal = con.execute("SELECT body FROM steps WHERE id=? AND kind='approval'", (sid,)).fetchone()
    if proposal and json.loads(proposal['body'] or '{}').get('action')=='revenue_mission':
        from . import revenue
        return revenue.approve(con,sid,action)
    if proposal and json.loads(proposal["body"] or "{}").get("action") == "renewal_schedule":
        from . import auto
        return auto.decide_renewal(con, sid, action)
    if not con.execute("UPDATE steps SET state=? WHERE id=? AND kind='approval' AND state='waiting'",
                       ("approved" if action == "approve" else "skipped", sid)).rowcount:
        con.commit()
        raise ValueError(f"step {sid} is not waiting")
    con.commit()
    s = con.execute("SELECT * FROM steps WHERE id=?", (sid,)).fetchone()
    b, t = json.loads(s["body"]), _thread_of(con, sid)
    try:
        if action == "skip":
            if b["action"] == "publish":
                db.feature("flows").decide_publish(con, b["workflow_id"], False)
        elif b["action"] == "people":
            for a in b.get("accounts") or [b]:
                o = b.get("product") or (con.execute("SELECT product FROM opps WHERE company_id=? ORDER BY score DESC",
                                                     (a["company_id"],)).fetchone() or [None])[0]
                b["found"] = b.get("found", 0) + people.find(con, a["company_id"], o, b["k"])
            n = b["found"]
            b["result"] = f"Found {n} contact{'s' * (n != 1)}" if n else "No new contacts found"
        elif b["action"] == "publish":
            try:
                db.feature("flows").decide_publish(con, b["workflow_id"], True)
                b["result"] = "Published"
            except ValueError:
                b["result"] = "Already decided on the Workflows page"
        elif b["action"] == "budget":
            start(t["id"], b["text"], b.get("extra"), force=True)
            b["result"] = "Carrying on"
        else:
            done = b["tasks"] = [_queue(con, d["company_id"], t["owner"], t, sid, d) for d in b["drafts"]]
            n, old = sum(bool(x) for x in done), sum(not x for x in done)
            b["result"] = f"{n} draft{'s' * (n != 1)} queued" + (f", {old} already in your inbox" if old else "")
    except Exception:
        traceback.print_exc()
        b["result"] = "It did not finish. Try again later."
    _set_body(con, sid, b)
    return con.execute("SELECT * FROM steps WHERE id=?", (sid,)).fetchone()


def _card_body(con, sid, kind):
    s = con.execute("SELECT * FROM steps WHERE id=? AND kind='card'", (sid,)).fetchone()
    b = json.loads(s["body"]) if s else {}
    if b.get("type") != kind:
        raise ValueError(f"step {sid} is not a {kind} card")
    return b


def save_segment(con, sid, owner):
    """Keep a list card under Saved lists. Saving the same card twice keeps one."""
    b = _card_body(con, sid, "accounts")
    con.execute("INSERT INTO segments(owner, name, items, step_id, created) VALUES(?,?,?,?,?) ON CONFLICT DO NOTHING",
                (owner, b["title"], json.dumps(b["items"]), sid, now()))
    seg = con.execute("SELECT id FROM segments WHERE step_id=?", (sid,)).fetchone()[0]
    _set_body(con, sid, b | {"saved": seg})
    return seg


def queue_draft(con, sid, owner):
    """One draft card into the seller's inbox. Queueing it twice keeps one task."""
    b = _card_body(con, sid, "draft")
    a = con.execute("SELECT id FROM companies WHERE uen=? ORDER BY in_crm DESC, id", (b["uen"],)).fetchone()
    if not a:
        raise ValueError(f"{b['name']} is not one of your accounts yet")
    t = _thread_of(con, sid)
    task = _queue(con, a["id"], owner, t, sid, b) or con.execute(
        "SELECT id FROM tasks WHERE company_id=? AND origin='agent' AND ref=?", (a["id"], f"thread:{t['id']}:{sid}")).fetchone()[0]
    _set_body(con, sid, b | {"queued": task})
    return task
