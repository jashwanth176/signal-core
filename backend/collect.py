"""Daily collectors: Google News RSS (5 languages) and GeBIZ contract awards.

Shortcut: Google News RSS is fine for a pilot; production swaps in a licensed news API.
GDELT comes in through graph.py and stock exchange filings (SGX included) through filings.py.
"""
import csv
import io
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import httpx

from .catalog import SINGTEL_GROUP

UA = {"User-Agent": "signal-research-bot/0.1"}
LOCALES = {  # lang -> (hl, gl, ceid)
    "en": ("en-SG", "SG", "SG:en"),
    "in": ("en-IN", "IN", "IN:en"),
    "vi": ("vi", "VN", "VN:vi"),
    "id": ("id", "ID", "ID:id"),
    "ja": ("ja", "JP", "JP:ja"),
}

# (query, locale). Local-language queries get translated by the tagger.
THEMES = [
    ('Singapore company expands to India', "en"),
    ('Singapore firm "new office" OR "new hub" Vietnam OR Indonesia OR India', "en"),
    ('Singapore logistics "new warehouse" OR "distribution centre"', "en"),
    ('Singapore manufacturer "new plant" OR "new factory"', "en"),
    ('Singapore "cyber attack" OR ransomware OR "data breach" company', "en"),
    ('Singapore appoints "chief information officer" OR CTO OR CISO', "en"),
    ('Singapore acquisition logistics OR manufacturing', "en"),
    ('Singapore company "digital transformation" OR "cloud migration"', "en"),
    ('Singapore "data centre" enterprise', "en"),
    ('Singapore tender network OR connectivity OR cybersecurity', "en"),
    ('Singapore company fleet tracking OR "smart warehouse" OR "smart factory"', "en"),
    ('DPDP rules compliance companies data localisation', "in"),
    ('Singapore company India plant OR facility OR expansion', "in"),
    ('ransomware attack India manufacturer OR logistics', "in"),
    ('doanh nghiệp Singapore mở rộng nhà máy', "vi"),
    ('tấn công mạng doanh nghiệp', "vi"),
    ('perusahaan Singapura ekspansi pabrik', "id"),
    ('serangan siber perusahaan Indonesia', "id"),
    ('シンガポール 企業 進出 拠点', "ja"),
    ('サイバー攻撃 製造業', "ja"),
    # PDPC decision pages render with JavaScript, so confirmed breaches come in through press coverage
    ('PDPC "financial penalty" OR "Protection Obligation" OR "data breach" fined', "en"),
]


def gnews(query, locale="en", when="14d", limit=15):
    """Fetch one Google News RSS search. `when` may be '14d', or None to pass after:/before: in query."""
    hl, gl, ceid = LOCALES[locale]
    q = f"{query} when:{when}" if when else query
    url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl={hl}&gl={gl}&ceid={ceid}"
    r = httpx.get(url, headers=UA, timeout=30, follow_redirects=True)
    r.raise_for_status()
    out = []
    for it in ET.fromstring(r.content).iter("item"):
        pub = it.findtext("pubDate")
        out.append({
            "source": "gnews:" + (it.findtext("source") or "?"),
            "url": it.findtext("link"),
            "title": it.findtext("title") or "",
            "body": re.sub(r"<[^>]+>", " ", it.findtext("description") or "").strip()[:500],
            "lang": locale if locale != "in" else "en",
            "country": gl,
            "published": parsedate_to_datetime(pub).date().isoformat() if pub else None,
        })
    out.sort(key=lambda x: x["published"] or "", reverse=True)
    return out[:limit]


def save_items(con, items):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seen = {r[0] for r in con.execute("SELECT lower(title) FROM items")}
    n = 0
    for i in items:
        t = i["title"].lower()
        if t in seen:
            continue
        seen.add(t)
        cur = con.execute(
            "INSERT INTO items(source,url,title,body,lang,country,published,fetched) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING",
            (i["source"], i["url"], i["title"], i["body"], i["lang"], i["country"], i["published"], now),
        )
        n += cur.rowcount
    con.commit()
    return n


def news(con, when="14d"):
    """Theme queries + one query per CRM account and per watched lookalike."""
    jobs = list(THEMES) + [(f'"{r["name"]}"', "en") for r in con.execute(
        "SELECT name FROM companies WHERE in_crm=1 OR watch IS NOT NULL")]
    total = 0
    for q, loc in jobs:
        try:
            total += save_items(con, gnews(q, loc, when))
        except (httpx.HTTPError, ET.ParseError) as e:
            print(f"  skip {q!r}: {e}")
        time.sleep(1)  # be polite
    return total


# ---------- GeBIZ awards (data.gov.sg open data) ----------
GEBIZ = "d_acde1106003906a75c3fa052592f2fcb"
CATEGORY = [  # first match wins
    ("cyber", r"cyber|security operation|\bsoc\b|penetration test|firewall|\bsiem\b|vulnerability|threat (intel|hunt)"),
    ("iot", r"\biot\b|internet of things|sensor|telematics|smart (building|lamp|nation)"),
    ("cloud", r"cloud|data cent|hosting|colocation|server farm"),
    ("connectivity", r"\bnetwork|\bwan\b|\blan\b|internet|broadband|connectivity|leased (line|circuit)|telecom|telephon"
                     r"|mobile (subscription|plan|line|data|phone)|fibre|fiber|wi-?fi|sd-?wan|voip|\bvoice\b"),
]
WORDS = dict(one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10)
TERM = re.compile(r"\b(\d+|" + "|".join(WORDS) + r")\s*\)?\s*(?:\(\d+\)\s*)?(years?|yrs?|months?)\b", re.I)


def category(desc):
    return next((c for c, rx in CATEGORY if re.search(rx, desc, re.I)), None)


def term_months(desc):
    """Contract term from tender text; telco contracts default to 36 months."""
    m = TERM.search(desc)
    if not m:
        return 36
    n = int(WORDS.get(m.group(1).lower(), m.group(1)))
    months = n * 12 if m.group(2).lower().startswith("y") else n
    return months if 6 <= months <= 120 else 36


def add_months(d, m):
    y, mo = divmod(d.month - 1 + m, 12)
    return date(d.year + y, mo + 1, min(d.day, 28))


def _get(url, timeout=60):
    """GET that waits out data.gov.sg's rate limit (429) instead of failing a long bulk load."""
    for wait in (5, 15, 30, 60, 120, 0):
        r = httpx.get(url, timeout=timeout, follow_redirects=True)
        if r.status_code != 429 or not wait:
            r.raise_for_status()
            return r
        time.sleep(wait)


def datagov_csv(dataset_id):
    """A whole data.gov.sg dataset as CSV text: initiate a download, poll for the file, fetch it."""
    api = f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}"
    _get(f"{api}/initiate-download")
    for _ in range(30):
        d = _get(f"{api}/poll-download").json()["data"]
        if d.get("url"):
            return _get(d["url"], timeout=600).content.decode("utf-8-sig")
        time.sleep(2)
    raise TimeoutError(f"data.gov.sg never produced {dataset_id}")


def gebiz(con):
    raw = datagov_csv(GEBIZ)
    n = 0
    for r in csv.DictReader(io.StringIO(raw)):
        cat = category(r["tender_description"])
        if not cat or r["supplier_name"] in ("Unknown", ""):
            continue
        d0 = datetime.strptime(r["award_date"], "%d/%m/%Y").date()
        tm = term_months(r["tender_description"])
        con.execute(
            "INSERT INTO awards VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(tender_no, supplier) DO UPDATE SET description=excluded.description, agency=excluded.agency, award_date=excluded.award_date, amount=excluded.amount, category=excluded.category, term_months=excluded.term_months, est_end=excluded.est_end",
            (r["tender_no"], r["tender_description"], r["agency"], d0.isoformat(), r["supplier_name"],
             float(r["awarded_amt"] or 0), cat, tm, add_months(d0, tm).isoformat()),
        )
        n += 1
    con.commit()
    return n


# ---------- ACRA new company registrations (data.gov.sg, one dataset per first letter) ----------
ACRA_COLLECTION = 2
# "X (Singapore) Pte Ltd", "X Asia Pacific Pte Ltd": how a foreign or regional group names its new Singapore entity
SUBSIDIARY = re.compile(r"\((SINGAPORE|S'?PORE|SG)\)|\bASIA\b|\bAPAC\b|ASIA PACIFIC|\bINTERNATIONAL\b|\bGLOBAL\b|\bREGIONAL\b", re.I)
# exempt private companies are owned by <= 20 individuals: small local firms, not groups
GROUP_TYPES = {"Private Company Limited by Shares", "Public Company Limited by Shares"}


def acra(con, days=90, per=400):
    """New Singapore entities of corporate groups in the last `days` days, saved as items for the tagger."""
    meta = httpx.get(f"https://api-production.data.gov.sg/v2/public/api/collections/{ACRA_COLLECTION}/metadata", timeout=30)
    ids = meta.json()["data"]["collectionMetadata"]["childDatasets"]
    since = date.fromordinal(date.today().toordinal() - days).isoformat()
    items = []
    for ds in ids:
        try:
            r = httpx.get("https://data.gov.sg/api/action/datastore_search", timeout=60,
                          params={"resource_id": ds, "sort": "uen_issue_date desc", "limit": per})
            rows = r.json()["result"]["records"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            print(f"  skip ACRA {ds}: {e}")
            continue
        for x in rows:
            if (x["uen_issue_date"] >= since and x["company_type_description"] in GROUP_TYPES
                    and SUBSIDIARY.search(x["entity_name"])):
                items.append({
                    "source": "acra", "url": f"https://data.gov.sg/datasets/{ds}/view?uen={x['uen']}",
                    "title": f"New Singapore company registered with ACRA: {x['entity_name']}",
                    "body": f"Registered {x['uen_issue_date']}. Activity: {x['primary_ssic_description']}. "
                            f"{x['primary_user_described_activity']}".replace(" na", "")[:400],
                    "lang": "en", "country": "SG", "published": x["uen_issue_date"]})
        time.sleep(1)
    return save_items(con, items)


STOP = set("tender provision services service supply delivery installation testing commissioning period years year "
           "months option options extend extension maintenance support invitation quotation contract framework "
           "agreement agency ministry singapore with from into upon based including the for and all new any one two "
           "three four five its per via are not".split())


def similar(a, b):
    """Word-overlap (Jaccard) of two tender descriptions; >= 0.2 means 'same thing bought again'."""
    A, B = ({w for w in re.findall(r"[a-z][a-z0-9-]{2,}", x.lower()) if w not in STOP} for x in (a, b))
    return len(A & B) / len(A | B) if A and B else 0.0


RENEWAL_SIM = 0.2


def is_singtel(supplier):
    return bool(re.search(SINGTEL_GROUP, supplier, re.I))


if __name__ == "__main__":
    assert term_months("MAINTENANCE FOR A PERIOD OF THREE (3) YEARS WITH OPTION OF 2 YEARS") == 36
    assert term_months("CyberArk Maintenance Support for 4 years") == 48
    assert term_months("Mobile plans for 24 months") == 24
    assert term_months("Supply of network switches") == 36
    assert category("Provision of Network Upgrading Services") == "connectivity"
    assert category("Security Operations Centre services") == "cyber"
    assert category("Mobile app development") is None
    assert is_singtel("NCS PTE. LTD.") and is_singtel("SINGAPORE TELECOMMUNICATIONS LIMITED")
    assert not is_singtel("NCS Pearson, Inc.")
    assert similar("Provision of WAN services for 3 years", "Provision of WAN services, option to extend") == 1.0
    assert similar("Mobile subscription plans", "Security operations centre") == 0.0
    print("collect ok")
