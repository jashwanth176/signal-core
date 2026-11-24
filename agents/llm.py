"""Model access and shared budget: Claude/Gemini research and Groq voice transcription.

Research calls use structured outputs (pydantic schema).
Evidence links are never written by the model: it cites signal ids and we render the URLs from the DB.
"""
import base64
import json
from pathlib import Path
import os
import time
import urllib.error
import urllib.request
import uuid
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import anthropic
from pydantic import BaseModel

from backend import db
from backend.catalog import INDUSTRY_FIT, PRODUCTS, SIGNALS

MODEL = os.environ.get("SIGNAL_MODEL", "claude-opus-5-5")
_client = None
_reserved_call = ContextVar('reserved_model_call', default=False)


class BudgetBlocked(RuntimeError):
    pass


def _reserve_budget():
    """Reserve one external model call, including speech, against the shared daily cap."""
    from .flows import CAP
    if CAP<=0:
        raise BudgetBlocked('Model research budget is zero')
    with db.connect() as con:
        r = con.execute("INSERT INTO kv(k,v) VALUES(?,'1') ON CONFLICT(k) DO UPDATE SET v=(kv.v::int+1)::text "
                        "WHERE kv.v::int<? RETURNING v",(calls_key(),CAP)).fetchone()
    if not r:
        raise BudgetBlocked(f"Today's {CAP} model calls are used; research waits for budget or seller retry")


def budgeted_ask(schema, system, user, **kw):
    """Reserve one daily call atomically before off-request work. ask() still owns every model adapter."""
    if not live():
        raise BudgetBlocked('Needs a model key; research is blocked, not completed')
    _reserve_budget()
    token = _reserved_call.set(True)
    try:
        return ask(schema,system,user,**kw)
    finally:
        _reserved_call.reset(token)


class SpeechNeedsKey(RuntimeError):
    pass


class SpeechUnavailable(RuntimeError):
    pass


def transcribe_audio(audio):
    """One bounded WAV chunk; the HTTP boundary validates audio before this model adapter."""
    key = os.environ.get('GROQ_API_KEY')
    if not key:
        raise SpeechNeedsKey('Voice transcription needs a server key. You can still type your instruction.')
    _reserve_budget()
    boundary = 'signal-' + uuid.uuid4().hex
    parts = []
    for name, value in [('model', 'whisper-large-v3-turbo'), ('response_format', 'json'), ('temperature', '0')]:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts += [f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="voice.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),
              audio, f'\r\n--{boundary}--\r\n'.encode()]
    request = urllib.request.Request('https://api.groq.com/openai/v1/audio/transcriptions',
        data=b''.join(parts), method='POST', headers={
            'Authorization': 'Bearer ' + key, 'Content-Type': 'multipart/form-data; boundary=' + boundary,
            'User-Agent': 'SignalSales/1.1'})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(32_001)
        if len(raw) > 32_000:
            raise ValueError('Oversized transcription')
        result = json.loads(raw)
        text = result.get('text')
        if not isinstance(text, str) or len(text) > 2000:
            raise ValueError('Invalid transcription')
        return text.strip()
    except (OSError, ValueError, TypeError, AttributeError) as error:
        if isinstance(error, urllib.error.HTTPError):
            error.close()
        raise SpeechUnavailable('Voice transcription is unavailable. Try again or type your instruction.') from None


def calls_key():
    """The kv key for today's model calls, a Singapore day: Signal's daily cap is counted against it."""
    return f"claude_calls:{datetime.now(timezone(timedelta(hours=8))).date()}"


def provider():
    """Which model answers: Claude when its key is set, else Gemini, else none (every caller handles "no key")."""
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "gemini" if os.environ.get("GEMINI_API_KEY") else None


def live():
    return provider() is not None


def calls_today():
    con = db.connect()
    try:
        return int(db.kv_get(con, calls_key(), 0))
    finally:
        con.close()


def _count():
    """Every model call is counted once (atomic: tagging runs in threads); the daily cap reads this count."""
    if not live():
        raise RuntimeError("no key")
    if _reserved_call.get():
        return
    con = db.connect()
    try:
        con.execute("INSERT INTO kv VALUES(?, '1') ON CONFLICT(k) DO UPDATE SET v=CAST(CAST(kv.v AS INTEGER) + 1 AS TEXT)",
                    (calls_key(),))
        con.commit()
    finally:
        con.close()


def client():
    global _client
    _count()
    _client = _client or anthropic.Anthropic()
    return _client


def ask(schema, system, user, effort="low", max_tokens=16000):
    if provider() == "gemini":
        return schema.model_validate(ask_json(system, user, schema.model_json_schema(), max_tokens=max_tokens))
    r = client().beta.messages.parse(
        model=MODEL,
        max_tokens=max_tokens,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # a safety decline is retried server-side on Anthropic's recommended model
        output_config={"effort": effort},
        output_format=schema,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if r.stop_reason == "refusal" or r.parsed_output is None:
        raise RuntimeError(f"model declined or returned nothing ({r.stop_reason}, req {r._request_id})")
    return r.parsed_output


def ask_json(system, user, schema, model=None, max_tokens=8000):
    """One structured call against a JSON schema dict; the answer as a dict. model: an id of the provider whose key
    is set (agentkit picks per provider), else that provider's default."""
    if provider() == "gemini":
        _count()
        r = _post(model or GEMINI_MODEL, {"systemInstruction": {"parts": [{"text": system}]},
                                 "contents": [{"role": "user", "parts": [{"text": user}]}],
                                 "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": max_tokens,
                                                      "responseJsonSchema": _inline(schema)}})
        text, _, finish = _parts(r)
        if not text:
            raise RuntimeError(f"model returned nothing ({finish})")
        return json.loads(text)
    r = client().beta.messages.create(
        model=model or MODEL, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
    if r.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(r.stop_reason)
    return json.loads(next(b.text for b in r.content if b.type == "text"))


# ---------------- Gemini (REST over urllib: no SDK) ----------------
GEMINI_MODEL = os.environ.get("SIGNAL_GEMINI_MODEL", "gemini-flash-latest")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"


def _post(model, body):
    """One generateContent request; retries rate limits and server errors twice."""
    req = urllib.request.Request(GEMINI_URL.format(model), data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"Gemini {e.code}: {e.read()[:300].decode(errors='replace')}") from None
            time.sleep(2 * (attempt + 1))


def _inline(schema):
    """Pydantic schemas point at $defs; send Gemini one tree with every $ref replaced by its definition."""
    defs = schema.get("$defs", {})

    def walk(x):
        if isinstance(x, dict):
            if "$ref" in x:
                return walk(defs[x["$ref"].rsplit("/", 1)[1]])
            # a "title" label is a string; a property named title is a schema and stays
            return {k: walk(v) for k, v in x.items() if k != "$defs" and not (k == "title" and isinstance(v, str))}
        return [walk(v) for v in x] if isinstance(x, list) else x
    return walk(schema)


def _parts(r):
    """(answer text, function-call parts, finish reason) of the first candidate; thought parts are not answer text."""
    c = (r.get("candidates") or [{}])[0]
    parts = (c.get("content") or {}).get("parts") or []
    text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
    return text, [p for p in parts if "functionCall" in p], c.get("finishReason") or (r.get("promptFeedback") or {}).get("blockReason")


TEXT_FILES = ("text/csv", "text/plain", "text/markdown")
MAX_FILE_TEXT = 200_000


def _file(b):
    """An attached file as ("text", words) or ("data", base64); gone files (a recycled Vercel instance) say so."""
    p = Path(b["path"])
    if not p.exists():
        return "text", f"(The file {b['name']} is no longer available. Ask the seller to attach it again.)"
    if b["media_type"] in TEXT_FILES:
        return "text", f"File {b['name']}:\n" + p.read_text(errors="replace")[:MAX_FILE_TEXT]
    return "data", base64.b64encode(p.read_bytes()).decode()


def _claude_blocks(messages):
    """Attached files as Claude content blocks: text files as text, PDFs as documents, pictures as images."""
    out = []
    for m in messages:
        if isinstance(m["content"], list) and any(b.get("type") == "file" for b in m["content"]):
            blocks = []
            for b in m["content"]:
                if b.get("type") != "file":
                    blocks.append(b)
                    continue
                kind, v = _file(b)
                src = {"type": "base64", "media_type": b["media_type"], "data": v}
                blocks.append({"type": "text", "text": v} if kind == "text" else
                              {"type": "document" if b["media_type"] == "application/pdf" else "image", "source": src})
            m = m | {"content": blocks}
        out.append(m)
    return out


def _gemini_contents(messages):
    names, out = {}, []
    for m in messages:
        blocks = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        parts = []
        for b in blocks:
            if b["type"] == "text" and b["text"]:
                parts.append({"text": b["text"]})
            elif b["type"] == "tool_use":
                names[b["id"]] = b["name"]
                parts.append({"functionCall": {"name": b["name"], "args": b["input"]},
                              **({"thoughtSignature": b["signature"]} if b.get("signature") else {})})
            elif b["type"] == "file":
                kind, v = _file(b)
                parts.append({"text": v} if kind == "text" else {"inlineData": {"mimeType": b["media_type"], "data": v}})
            elif b["type"] == "tool_result":
                c = b.get("content")
                c = c if isinstance(c, str) else "".join(x.get("text", "") for x in c or [])
                parts.append({"functionResponse": {"name": names.get(b["tool_use_id"], "tool"), "response": {"result": c}}})
        if parts:
            out.append({"role": "model" if m["role"] == "assistant" else "user", "parts": parts})
    return out


def _gemini_turn(system, messages, tools):
    _count()
    r = _post(GEMINI_MODEL, {"systemInstruction": {"parts": [{"text": system}]}, "contents": _gemini_contents(messages),
                             "tools": [{"functionDeclarations": [{"name": t["name"], "description": t.get("description", ""),
                                                                  "parametersJsonSchema": _inline(t["input_schema"])} for t in tools]}],
                             "generationConfig": {"maxOutputTokens": 16000}})
    text, calls, finish = _parts(r)
    content = [{"type": "text", "text": text}] if text else []
    content += [{"type": "tool_use", "id": "g" + uuid.uuid4().hex[:12], "name": p["functionCall"]["name"],
                 "input": p["functionCall"].get("args") or {}, **({"signature": p["thoughtSignature"]} if p.get("thoughtSignature") else {})}
                for p in calls]
    return {"stop_reason": "tool_use" if calls else "max_tokens" if finish == "MAX_TOKENS" else "end_turn", "content": content}


# ---------------- tagger ----------------
Signal = Literal[tuple(SIGNALS) + ("none",)]
Product = Literal[tuple(PRODUCTS) + ("none",)]


class Related(BaseModel):
    name: str
    kind: Literal["subsidiary", "parent", "supplier", "customer", "partner", "acquirer", "acquired"]


class Tag(BaseModel):
    id: int
    signal: Signal
    company: str
    industry: Literal[tuple(INDUSTRY_FIT)]
    hq_country: str
    size: Literal["large", "mid", "small", "unknown"]
    country: str
    product_hint: Product
    confidence: float
    summary_en: str
    related: list[Related]
    direction: Literal["grow", "shrink", "rival", "none"]
    part: str


class Tags(BaseModel):
    tags: list[Tag]


TAG_SYSTEM = f"""You tag news for Singtel's enterprise sales team ({", ".join(PRODUCTS.values())}).
For each item decide whether it reveals a BUYING MOMENT for a specific organisation, using exactly one signal type:
{json.dumps(SIGNALS, indent=1)}
Use "none" for consumer news, opinion, stock tips, generic market commentary, or anything without a clear buying moment.
A buying moment also covers a company starting something that will need a network: a new service or business line that
runs on technology, or a strategic pivot (digital_transformation: a hospital starting remote surgery or tele-consults, a
bank launching a digital bank, a retailer opening an online marketplace), and a new department or team
(hiring_tech: a new AI office, a new regional shared-services centre).
The sales team also needs to know when NOT to call, so tag these too, with the type of the thing that is changing:
- direction "shrink": the company is cutting back. It sells, closes or winds up a unit (merger_acquisition), closes a
  site (new_facility), leaves a country (new_market_expansion), lays off staff or folds teams together (hiring_tech), or
  is itself being wound up or taken over (merger_acquisition).
- direction "rival": the company has just signed, renewed or extended a network, cloud, security, data centre or IoT
  contract with a supplier other than Singtel (contract_ending, product_hint = that product, the supplier in `related`).
Items titled "New Singapore company registered with ACRA" are new Singapore entities: tag new_market_expansion (country SG)
only when the name shows a known foreign or regional group setting up in Singapore, with company = that group's English
name (e.g. "Toyota Tsusho (Singapore) Pte Ltd" -> Toyota Tsusho); otherwise "none".
Items titled "Stock exchange filing" are a listed company's own disclosure to NSE or BSE (India), SGX (Singapore) or ASX
(Australia): company = the filer named in the title. Most are paperwork. Tag one only when the filer itself starts, buys
or wins something that will operate: a new operating subsidiary, office, plant or data centre in another country
(new_market_expansion, country = where it will operate) or at home (new_facility); buying an operating business or
forming a joint venture, also when done through a subsidiary (merger_acquisition); a new CIO, CTO, CISO or head of IT
(leadership_change); or when it sells, closes, liquidates or strikes off an operating business, site or subsidiary
(direction "shrink"; a dormant shell being struck off is "none"). Otherwise "none": any other appointment (a CEO,
directors, finance or investment officers), money put into a subsidiary or holding company it already owns, refinancing
or loans, transfers inside the group, a passive stake or a larger stake in a company it already part-owns, an agreement
to explore or cooperate, results, share dealings, or the sale of a property or a financial holding.
data_regulation may have an empty company when it is a rule for a whole market or sector.
Fields:
- company: the organisation with the buying moment, its full English name, not an abbreviation (Singapore Post, not SingPost; no "Pte Ltd"). Empty if none.
- industry/size/hq_country: your best knowledge of that company (ISO-2 country, "" if unknown).
- country: ISO-2 country where the activity happens (e.g. the new factory's country).
- product_hint: only for tender_published, the product the tender is for; otherwise "none".
- confidence: 0-1 that the signal type is right AND it matters for an enterprise telco.
- summary_en: one plain English sentence (translate if the item is not English).
- related: other named companies in the item with their relation to `company` (acquirer, supplier, partner...). Empty list if none.
- direction: "grow" when the company is adding something, "shrink" or "rival" as described above, otherwise "none".
- part: the business line, subsidiary or site the item is about, in a few words ("Cambodia energy business", "rail
  operations"); "" when it concerns the whole company.
Return one tag per input item, same id."""


def tag(items):
    """items: list of dict(id, title, body, lang, country, published). Returns list[Tag]."""
    batches = [items[i:i + 25] for i in range(0, len(items), 25)]

    def one(batch):
        lines = "\n".join(json.dumps({k: it[k] for k in ("id", "title", "body", "lang", "country", "published")},
                                     ensure_ascii=False) for it in batch)
        return ask(Tags, TAG_SYSTEM, f"Items (JSON lines):\n{lines}").tags

    with ThreadPoolExecutor(4) as ex:
        return [t for ts in ex.map(one, batches) for t in ts]


# ---------------- action card ----------------
class WhyNow(BaseModel):
    signal_id: int
    line: str


class Card(BaseModel):
    headline: str
    why_now: list[WhyNow]
    customer_goal: str
    pitch: str
    contact_role: str
    opener: str
    email_subject: str
    email_body: str
    risk: str


CARD_SYSTEM = """You write action cards for a Singtel enterprise salesperson. Be concrete, short, and honest.
- why_now: the 2-3 strongest signals, each one line, citing its signal_id from the input. Never invent facts or links.
- customer_goal: what the customer is trying to achieve, in their terms.
- pitch: 2-3 sentences tying the Singtel product to that goal. Cite a deal from case_studies (won in Signal, or a published Singtel announcement) only if one is given; never invent a customer win.
- contact_role: a job title to approach (no personal names).
- opener: something useful to give the customer first (an insight, benchmark or intro), not a product pitch.
- email_body: under 120 words, plain text, warm and specific, signed with the name in "from", then "Singtel Enterprise". If a warm path exists, use it.
- risk: one line on what could make this opportunity wrong."""


def write_card(ctx):
    return ask(Card, CARD_SYSTEM, json.dumps(ctx, ensure_ascii=False, default=str), effort="medium")


# ---------------- customer benchmark report ----------------
class Point(BaseModel):
    text: str
    signal_ids: list[int]


class Section(BaseModel):
    heading: str
    points: list[Point]


class Report(BaseModel):
    title: str
    intro: str
    sections: list[Section]
    questions: list[str]


REPORT_SYSTEM = """You write a short, genuinely useful benchmark briefing a Singtel salesperson can GIVE to a customer
before any pitch: how the customer's peers are handling what the peer signals show. Pick the 2-3 themes the signals
actually cover, one section each. Use only the peer signals provided; cite their signal_ids on every point. Neutral
tone, no Singtel product names. 2-4 points per section. End with 3 discussion questions for the customer."""


def write_report(ctx):
    return ask(Report, REPORT_SYSTEM, json.dumps(ctx, ensure_ascii=False, default=str), effort="medium")


# ---------------- back-test: harvest past Singtel wins from headlines ----------------
class Win(BaseModel):
    id: int
    is_win: bool
    customer: str
    product: Product
    date: str


class Wins(BaseModel):
    wins: list[Win]


WIN_SYSTEM = """Each item is a news headline mentioning Singtel or NCS. Mark is_win=true only when it reports that a named
enterprise or government customer (not a consumer, not a telco partner, not Singtel's own investment) chose / signed /
deployed a Singtel or NCS solution. customer: that customer's name. product: closest product key. date: item date (YYYY-MM-DD).
One output per input id."""


def harvest(items):
    lines = "\n".join(json.dumps({"id": i["id"], "title": i["title"], "date": i["published"]}, ensure_ascii=False)
                      for i in items)
    return ask(Wins, WIN_SYSTEM, lines).wins


# ---------------- account research brief (pages from the company's own site) ----------------
class Fact(BaseModel):
    text: str
    source: int  # index into the pages given


class Brief(BaseModel):
    what_they_do: str
    priorities: list[Fact]
    footprint: list[Fact]
    angles: list[Fact]
    questions: list[str]


BRIEF_SYSTEM = """You research one account for a Singtel enterprise seller, using ONLY the numbered pages from the
company's own website. Every fact cites the page number it came from. Never invent numbers, names or plans.
- what_they_do: two plain sentences.
- priorities: 3-5 things the company says it is pushing (expansion, digital, sustainability, new sites...).
- footprint: countries, sites, fleets, systems or partners named on the pages (what a network or security team must cover).
- angles: 2-4 places where one of these would help a stated priority: {", ".join(PRODUCTS.values())}.
  Tie each to the page that shows the need. No product pitch language.
- questions: 3 sharp discovery questions a seller could ask in the first meeting."""


def write_brief(name, pages):
    body = "\n\n".join(f"[page {i}] {u}\n{t}" for i, (u, t) in enumerate(pages))
    b = ask(Brief, BRIEF_SYSTEM, f"Account: {name}\n\n{body}", effort="medium")
    if not b.questions:  # the model left the last field empty on 2 of 3 accounts; one retry fills it
        b = ask(Brief, BRIEF_SYSTEM + "\nquestions must hold exactly 3 items.", f"Account: {name}\n\n{body}", effort="medium")
    return b


# ---------------- ask about an account ----------------
class Cited(BaseModel):
    text: str
    ids: list[int]  # fact ids from the input


class Answer(BaseModel):
    answer: str
    points: list[Cited]


ANSWER_SYSTEM = """You answer a Singtel enterprise seller's question about one account, using ONLY the numbered facts given.
- answer: two or three plain sentences. If the facts do not answer the question, say so plainly and say what is missing.
- points: up to 4 supporting points, each citing the fact ids it rests on. A point with no fact behind it is not allowed.
- Never invent names, numbers, dates or plans. Never write URLs or email addresses; the app links the sources."""


def answer(question, ctx):
    return ask(Answer, ANSWER_SYSTEM, json.dumps({"question": question, **ctx}, ensure_ascii=False, default=str),
               effort="medium")


# ---------------- account plan ----------------
from backend.catalog import PLAN_SECTIONS  # noqa: E402


class PlanSec(BaseModel):
    key: Literal[tuple(k for k, _ in PLAN_SECTIONS)]
    text: str
    ids: list[int]


class PlanOut(BaseModel):
    sections: list[PlanSec]


PLAN_SYSTEM = f"""You keep a one-page account plan for a Singtel enterprise seller, using ONLY the numbered facts and the
buying committee given. Write one entry per section, in this order: {", ".join(f"{k} ({h})" for k, h in PLAN_SECTIONS)}.
- Each text is 2-4 plain sentences a sales manager would read in a minute. Concrete: names of contracts, dates, roles.
- ids: the fact ids each section rests on. The committee section is written from the committee given, ids may be empty.
- next: 2-3 dated or ordered actions, smallest first. risks: what could lose or stall the deal.
- If the facts are thin, say what is unknown instead of guessing. Never invent names, numbers or plans.
- Never write URLs or email addresses; the app links the sources."""


def write_plan(ctx):
    return ask(PlanOut, PLAN_SYSTEM, json.dumps(ctx, ensure_ascii=False, default=str), effort="medium")


# ---------------- overnight drafts for play tasks ----------------
class Draft(BaseModel):
    to_role: str
    subject: str
    body: str
    ids: list[int]  # fact ids the email rests on


DRAFT_SYSTEM = """You draft one short email for a Singtel enterprise seller. The task says which play fired and why.
- body: under 120 words, plain text, warm and specific. Give something useful first (an insight, a benchmark, an offer
  to brief their team), then one light ask. Signed with the name in "from", then "Singtel Enterprise".
- Greet the person by first name only if a name is given in "to"; otherwise write "Hi," and never invent a name.
- to_role: the job title the email is for. ids: the fact ids the email rests on.
- Use only the facts given. Never write URLs or email addresses."""


def write_draft(ctx):
    return ask(Draft, DRAFT_SYSTEM, json.dumps(ctx, ensure_ascii=False, default=str), effort="low", max_tokens=4000)


# ---------------- Ask Signal: the agent thread (tool use, a faster model than cards and plans) ----------------
AGENT_MODEL = os.environ.get("SIGNAL_AGENT_MODEL", "claude-sonnet-5-5")

AGENT_SYSTEM = """You are Signal, the agent of a Singtel enterprise seller in Singapore. Work the seller's question
with your tools, step by step, then answer it.
- Look things up before you answer. Never invent company names, UENs, people, numbers or dates: every claim comes from a
  tool result, and you say where it came from (registry, GeBIZ, news, CRM, Apollo); when every point shares one
  source, name it once, not after each bullet.
- Get a company's UEN from find_company before any tool that needs one, unless the seller attached the company (its UEN
  is then in their message). If a tool says not found, say so; do not guess.
- Files the seller attaches (CSV lists, PDFs, pictures) are part of their message: read them and work from them.
- Show what you found as cards when it helps: show_accounts for a list of companies, each with one line on why it fits
  and its source; show_brief for sourced points about one company; show_draft for an email.
- When a real choice is the seller's (which product, how many accounts, which role), ask with ask_seller (2-4 options)
  and end your turn; their answer is their next message.
- Anything that spends Apollo credits (finding contacts) or queues several drafts goes through request_people_lookup or
  request_queue_drafts, workflows through build_workflow's publish card. Those only ask the seller; never say it is done
  until the seller approves. End your turn after posting one. Nothing ever sends email.
- Never write email addresses. Drafts: under 120 words, plain text, warm and specific, "Hi," unless a tool gave you the
  person's name, signed with the seller's name (given below), then "Singtel Enterprise".
- Answer in short markdown: a one-line lead, then "- " bullets with **bold** names where a list helps, and one closing
  line on what you can do next. No headings, no tables."""


def agent_turn(system, messages, tools):
    """One model round of the Ask Signal thread: {"stop_reason", "content": [block dicts]}. Blocks come back as plain
    dicts (text, tool_use, and thinking blocks, which the loop must send back unchanged) so a thread can be stored."""
    if provider() == "gemini":
        return _gemini_turn(system, messages, tools)
    r = client().beta.messages.create(
        model=AGENT_MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # a safety decline is retried server-side on Anthropic's recommended model
        output_config={"effort": "medium"},
        cache_control={"type": "ephemeral"},  # each round resends the whole thread; the prefix is read from cache
        system=system,
        tools=tools,
        messages=_claude_blocks(messages),
    )
    return {"stop_reason": r.stop_reason,
            "content": [b.model_dump(mode="json", exclude_none=True) for b in r.content if b.type != "fallback"]}


# ---------------- outreach follow-ups (autopilot) ----------------
class Followup(BaseModel):
    body: str


FOLLOWUP_SYSTEM = """You write one short follow-up for a Singtel enterprise seller whose earlier email has had no reply.
- body: under 70 words, plain text, polite and specific to the earlier email. Add one new useful thing or make the ask
  smaller; never guilt the reader and never write "just checking in" or "bumping this". The last follow-up says it is
  the last note.
- Greet the person by first name from "to". Sign with the seller's name from "from", then "Singtel Enterprise".
- The earlier email is quoted below your text by the app; do not repeat it.
- Follow "instructions" when given (the seller's own words for this play) and use "company_notes" for how Singtel sells.
- Use only what is given. Never write URLs or email addresses."""


def write_followup(ctx):
    return ask(Followup, FOLLOWUP_SYSTEM, json.dumps(ctx, ensure_ascii=False, default=str), effort="low", max_tokens=2000)


# ---------------- workflows: an AI question whose answer picks the branch ----------------
class Pick(BaseModel):
    choice: str  # exactly one of the options given
    why: str
    ids: list[int]  # fact ids the answer rests on


PICK_SYSTEM = """You answer one question about one account for a Singtel enterprise seller's workflow. The answer picks
which branch the workflow takes next.
- choice: exactly one of the options given, copied character for character.
- why: one plain sentence saying what in the facts decided it. If the facts do not settle it, pick the most cautious
  option and say what is missing.
- ids: the fact ids the answer rests on. Use only the facts given; never invent names, numbers or dates."""


def pick_branch(question, options, ctx):
    return ask(Pick, PICK_SYSTEM, json.dumps({"question": question, "options": options, **ctx}, ensure_ascii=False,
                                             default=str), effort="low", max_tokens=2000)


# ---------------- outreach: a drafted answer to a contact's reply (the seller edits and sends it) ----------------
ANSWER_SYSTEM = """A contact answered a Singtel enterprise seller's email. Draft the seller's answer, which the seller
will edit and send by hand.
- body: under 90 words, plain text. Respond to what the contact actually wrote: answer their question, accept or offer
  a time, or thank them and close politely if they declined. Offer one clear next step.
- Greet the person by first name from "to". Sign with the seller's name from "from", then "Singtel Enterprise".
- Follow "instructions" when given (the seller's own words for this play) and use "company_notes" for how Singtel sells.
- Use only what is given. Never invent prices, dates, names or commitments; never write URLs or email addresses."""


def write_answer(ctx):
    return ask(Followup, ANSWER_SYSTEM, json.dumps(ctx, ensure_ascii=False, default=str), effort="low", max_tokens=2000)
