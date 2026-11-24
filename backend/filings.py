"""Stock exchange filings as items for the tagger: NSE and BSE (India), SGX (Singapore), ASX (Australia).

The four exchanges publish about 2,000 notices a day and nearly all are routine, so a filing is kept only when its text
names the other country of a Singapore corridor and an expansion action, and no exclusion term (RULES). Kept filings
get the first pages of their PDF read, so the tagger sees what the company actually disclosed.

The endpoints are the ones each exchange's own page calls. They refuse non-browser clients (SGX answers Access Denied),
so requests go through curl_cffi with a Chrome fingerprint. curl_cffi and pypdf are imported inside the functions: the
serverless copy installs neither and never collects.

SGX is also kept whole: graph.sgx stores every announcement's headline in the registry graph, `feed` turns the ones
CRM accounts and watched companies file themselves into items, and `read` fetches one filing's text for an agent.
"""
import codecs
import csv
import html
import io
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlsplit

from . import db
from .collect import save_items

PAUSE = 0.8  # seconds between requests to one exchange
PREFIX = "Stock exchange filing"  # the tagger prompt (llm.TAG_SYSTEM) names this title prefix
COMPANY = re.compile(r"^Stock exchange filing \(\w+, [\d-]+\): (.*?): ")  # the filer, read back from an item title

SINGAPORE = r"singapore|pte\.? ltd"
INDIA = (r"india|mumbai|bengaluru|bangalore|chennai|hyderabad|pune|delhi|gurugram|noida|kolkata|ahmedabad|tamil nadu"
         r"|maharashtra|karnataka|gujarat|telangana|andhra pradesh|uttar pradesh|haryana|chhattisgarh|odisha|kerala"
         r"|west bengal|rajasthan|madhya pradesh")
AUSTRALIA = r"australia|sydney|melbourne|brisbane|perth|adelaide|canberra|queensland|new south wales|pty\.? ltd"
GONE = r"winding up|dissolution|dissolv|strike off|struck off|deregist"
RULES = {  # exchange -> [(places, actions, exclude)]: a place and an action must appear, an exclude term vetoes
    "nse": [(SINGAPORE,
             r"subsidiar|incorporat|joint venture|acqui|investment|stake|office|expansion|\bmou\b|partner|branch",
             GONE + r"|investor meet|analyst|roadshow|con\. call|conference|earnings call|regulation 31"
                    r"|substantial acquisition of shares")],
    "sgx": [(INDIA,
             r"subsidiar|incorporat|joint venture|acqui|investment|stake|office|plant|facility|expansion|\bmou\b|partner"
             r"|data cent(?:re|er)",
             GONE + r"|disposal|divest|differences between financial forecasts"),
            (AUSTRALIA,
             r"subsidiar|incorporat|joint venture|acqui|investment|office|plant|facility|expansion|\bmou\b|partner"
             r"|data cent(?:re|er)",
             GONE + r"|disposal|divest|differences between financial forecasts|investment forum|conference|webinar")],
    "asx": [(SINGAPORE,
             r"subsidiar|incorporat|joint venture|acqui|investment|office|expansion|expand|\bmou\b|memorandum|partnership"
             r"|branch|establish|launch|data cent(?:re|er)",
             GONE + r"|substantial holder|substantial shareholder|form 60[345]|director's interest|appendix 3y"
                    r"|appendix 2a|disposal|divest|investor presentation|roadshow|conference|webinar")],
}
RULES["bse"] = RULES["nse"]


def hits(exchange, text, company=""):
    """The place and action terms that keep a filing, or [] when it is routine. An action word that is only part of the
    filer's own name ("XYZ Investments Ltd") does not count."""
    t, co = text.lower(), company.lower()
    for places, actions, exclude in RULES[exchange]:
        if re.search(exclude, t):
            continue
        p, a = set(re.findall(places, t)), {w for w in re.findall(actions, t) if w not in co}
        if p and a:
            return sorted(p) + sorted(a)
    return []


def excerpt(doc, terms, width=900):
    """The stretch of a document around the first matched term: the disclosure, not the letterhead."""
    at = min((i for i in (doc.lower().find(w) for w in terms) if i >= 0), default=0)
    return doc[max(0, at - width // 3):][:width]


def item(exchange, country, company, headline, summary, doc, url, day, terms):
    company, headline = (" ".join((x or "").split()) for x in (company, headline))
    return {"source": exchange, "url": url, "lang": "en", "country": country, "published": day, "company": company,
            # the day keeps the title unique: collect.save_items drops repeated titles, and NSE headlines are
            # category labels ("Updates")
            "title": f"{PREFIX} ({exchange.upper()}, {day}): {company}: {headline}"[:300],
            "body": f"{' '.join(summary.split())[:400]} {excerpt(doc, terms)}".strip()}


# ---------- HTTP ----------
def _session(referer):
    from curl_cffi import requests
    s = requests.Session(impersonate="chrome")
    s.headers.update({"Referer": referer, "Accept": "application/json, text/plain, */*"})
    return s


def _get(s, url, timeout=60, **kw):
    """GET with the polite pause; a network error, 429 or 5xx is retried twice."""
    for wait in (20, 40, 0):
        try:
            r = s.get(url, timeout=timeout, **kw)
        except Exception:  # curl_cffi's errors: the import is lazy, so its classes are not in scope here
            if not wait:
                raise
            r = None
        time.sleep(PAUSE)
        if r is not None and not (wait and (r.status_code == 429 or r.status_code >= 500)):
            r.raise_for_status()
            return r
        time.sleep(wait)


def pdf_text(s, url, pages=3):
    """Text of the first pages of the PDF at url, or '' when it cannot be fetched or read (scans have no text)."""
    from pypdf import PdfReader
    from pypdf.errors import DependencyError
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # malformed-PDF warnings are noise
    try:
        raw = _get(s, url, timeout=120).content
        if raw[:4] != b"%PDF":
            return ""
        return " ".join(" ".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(raw)).pages[:pages]).split())
    except DependencyError:  # ASX PDFs are AES-protected: without pypdf[crypto] every one would read as empty
        raise
    except Exception:
        return ""


# ---------- NSE: one call returns every announcement in the window ----------
NSE_PAGE = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"


def nse(start, end):
    s = _session(NSE_PAGE)
    s.get(NSE_PAGE, timeout=60)  # sets the cookies the API asks for
    rows = _get(s, "https://www.nseindia.com/api/corporate-announcements", timeout=180, params={
        "index": "equities", "from_date": f"{start:%d-%m-%Y}", "to_date": f"{end:%d-%m-%Y}"}).json()
    for r in rows if isinstance(rows, list) else []:
        company, summary = r.get("sm_name") or "", r.get("attchmntText") or ""
        terms = hits("nse", f"{r.get('desc')} {summary}", company)
        if not terms:
            continue
        pdf = (r.get("attchmntFile") or "").strip()
        pdf = "" if pdf in ("-", "NA") else pdf  # NSE writes "-" when there is no attachment
        yield item("nse", "IN", company, r.get("desc"), summary, pdf_text(s, pdf) if pdf else "",
                   pdf or f"https://www.nseindia.com/get-quotes/equity?symbol={r.get('symbol')}#seq={r.get('seq_id')}",
                   (r.get("sort_date") or "")[:10], terms)


# ---------- BSE: same-day queries only, 50 rows a page ----------
BSE_FILES = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/"


def bse_page(s, day, page):
    """One page of a day's announcements and the day's total. BSE answers about one request in a hundred with "Bad
    Request" or an HTML page and the same request works a moment later, so it gets three tries."""
    for wait in (3, 10, 0):
        try:
            data = _get(s, "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w", params={
                "pageno": page, "strCat": "-1", "subcategory": "-1", "strPrevDate": day, "strToDate": day,
                "strScrip": "", "strSearch": "P", "strType": "C"}).json()
            return data["Table"], int((data.get("Table1") or [{}])[0].get("ROWCNT") or 0)
        except Exception:
            if not wait:
                raise
            time.sleep(wait)


def bse_items(s, rows):
    for r in rows:
        company, summary = (r.get("SLONGNAME") or "").removesuffix("-$"), r.get("NEWSSUB") or ""  # "-$": a BSE marker
        headline = r.get("HEADLINE") or summary
        terms = hits("bse", f"{headline} {summary}", company)
        if not terms:
            continue
        pdf = BSE_FILES + r["ATTACHMENTNAME"] if r.get("ATTACHMENTNAME") else ""
        subject = re.sub(r"^.*? - \d{6} - ", "", summary)  # NEWSSUB opens with "<company> - <scrip code> - "
        yield item("bse", "IN", company, subject or headline, headline, pdf_text(s, pdf) if pdf else "",
                   pdf or f"https://www.bseindia.com/corporates/ann.html#{r.get('NEWSID')}",
                   (r.get("NEWS_DT") or "")[:10], terms)


def bse(start, end):
    s, later = _session("https://www.bseindia.com/"), []
    for n in range((end - start).days + 1):
        day = f"{start + timedelta(days=n):%Y%m%d}"
        for page in range(1, 81):
            try:
                rows, total = bse_page(s, day, page)
            except Exception:
                later.append((day, page))  # asked again at the end: a bad spell passes within minutes
                if len(later) > 8:  # blocked, not flaky
                    raise
                continue
            yield from bse_items(s, rows)
            if not rows or page * 50 >= total:
                break
    for day, page in later:
        try:
            yield from bse_items(s, bse_page(s, day, page)[0])
        except Exception as e:
            print(f"  skip bse {day} page {page}: {type(e).__name__}")


# ---------- SGX: titles never name a country, so likely titles get their detail page and PDF read ----------
SGX_TITLES = (r"subsidiar|incorporat|acqui|joint venture|memorandum|\bmou\b|investment|establish|expansion|partnership"
              r"|collaboration|agreement|new office|data cent(?:re|er)|plant|facility")
SGT = timezone(timedelta(hours=8))
SGX_BODY = re.compile(r"Description \(Please provide a detailed description of the event in the box below\)(.*?)"
                      r"(?:Attachments|Total size|$)")


def sgx_session():
    """A session, the announcements API and its header. The token is what sgx.com itself fetches on every visit: a CMS
    value, ROT13-decoded."""
    s = _session("https://www.sgx.com/")
    cfg = _get(s, "https://www.sgx.com/config/appconfig.json").json()
    api, cms = cfg["endpoints"]["ANNOUNCEMENTS_API_URL"], cfg["endpoints"]["CMS_API_URL"]
    token = _get(s, f"{cms}/?queryId={cfg['CMS_VERSION']}:we_chat_qr_validator").json()["data"]["qrValidator"]
    return s, api, {"authorizationToken": codecs.decode(token, "rot13")}


def sgx_rows(s, api, auth, start, end):
    """Every announcement broadcast from `start` to `end`, as SGX lists them (about 100 a day)."""
    span = {"periodstart": f"{start:%Y%m%d}_000000", "periodend": f"{end:%Y%m%d}_235959"}
    total = int(_get(s, api + "count", params=span, headers=auth).json().get("data") or 0)
    for page in range((total + 249) // 250):
        yield from _get(s, api, params={**span, "pagestart": page, "pagesize": 250}, headers=auth).json().get("data") or []


def sgx_day(r):
    ms = r.get("broadcast_date_time") or r.get("submission_date_time") or 0
    return datetime.fromtimestamp(ms / 1000, tz=SGT)


def sgx_read(s, url):
    """What an announcement says: the description on its page, and the first pages of its first PDF."""
    page_html = _get(s, url).text
    text = " ".join(html.unescape(re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", page_html,
                                         flags=re.S | re.I)).split())
    m = SGX_BODY.search(text)
    pdfs = re.findall(r'href="([^"]+\.pdf[^"]*)"', page_html, re.I)
    link = html.unescape(pdfs[0]) if pdfs else ""
    doc = pdf_text(s, link if link.startswith("http") else "https://links.sgx.com" + link) if link else ""
    return (m.group(1) if m else text).strip(), doc


def sgx(start, end):
    s, api, auth = sgx_session()
    for r in sgx_rows(s, api, auth, start, end):
        title, issuers = r.get("title") or "", r.get("issuers") or []
        # foreign groups that only list bonds here (no Singapore ISIN) are not Singapore companies
        if not (any((i.get("isin_code") or "").upper().startswith("SG") for i in issuers)
                and re.search(SGX_TITLES, title.lower()) and r.get("url")):
            continue
        company = r.get("issuer_name") or issuers[0].get("issuer_name") or ""
        summary, doc = sgx_read(s, r["url"])
        terms = hits("sgx", f"{title} {summary} {doc}"[:4000], company)
        if terms:
            yield item("sgx", "SG", company, title, summary, doc, r["url"], sgx_day(r).date().isoformat(), terms)


# ---------- accounts' own SGX announcements, out of the archive graph.sgx keeps ----------
FEED = r"general announcement|asset acquisitions|tender/ acquisition|takeover|placement"
TECH_ROLE = r"information|technology|digital|security|\bdata\b|\bc[it]o\b|\bciso\b"
# half of all "General Announcement"s are paperwork; these headline words mark it (checked on 90 days of 13 accounts)
FEED_NOISE = (r"business updates?|investor presentation|conference call|minutes of|general meeting"
              r"|composition of the board|director|company secretary|cessation|liquidat|\bnotes\b|perpetual"
              r"|debt issuance|redemption|tender offer|trust deed|quer(?:y|ies)|response|notice of"
              r"|holding announcement|share plan|allotment|financial results|legal proceedings|arbitration|patent")


def feed_worthy(category, title):
    """Can this kind of announcement carry a buying moment? Buy-backs, share dealings, coupons, dividends, results and
    meetings never do; an appointment only when it is a technology role; a general announcement unless its headline
    marks it as paperwork (a note issue, board changes, an investor presentation)."""
    c, t = (category or "").lower(), (title or "").lower()
    if "appointment" in c:
        return bool(re.search(TECH_ROLE, t))
    return bool(re.search(FEED, c)) and not re.search(FEED_NOISE, t)


def feed(app, g, days=30, per=10, most=150):
    """Each CRM account's and watched company's own SGX announcements of the last `days` days, read (page and PDF)
    and saved as items for the tagger. The corridor rules above only keep filings that name India or Australia; for a
    company we already track, a deal anywhere or a new technology chief is a buying moment too. An announcement
    already saved is not read again, and the title is the one `sgx` writes, so the two never save one filing twice.
    At most `per` a company and `most` a run, newest first: a long watch list catches up over the next runs."""
    from . import kg
    since = (date.today() - timedelta(days=days)).isoformat()
    have = {r["url"] for r in app.execute("SELECT url FROM items WHERE source='sgx'")}
    todo = {}
    for c in app.execute("SELECT uen FROM companies WHERE (in_crm=1 OR watch IS NOT NULL) AND uen IS NOT NULL").fetchall():
        fresh = [f for f in kg.filings(g, c["uen"], since, 200)
                 if f["url"] not in have and feed_worthy(f["category"], f["title"])]
        todo.update({f["url"]: f for f in fresh[:per]})
    items, failed, s = [], 0, _session("https://www.sgx.com/") if todo else None
    for f in sorted(todo.values(), key=lambda f: f["at"], reverse=True)[:most]:
        try:
            summary, doc = sgx_read(s, f["url"])
        except Exception as e:
            failed += 1
            if failed == 3:  # SGX is down or blocking: the rest waits for the next run
                print(f"  skip sgx feed: {type(e).__name__}: {e}")
                break
            continue
        # a press release repeats its headline above the first paragraph: start there, not at the letterhead
        head = f["title"].split("::")[-1].strip().lower()[:40]
        doc = doc[max(0, doc.lower().find(head)) if head else 0:]
        items.append(item("sgx", "SG", f["issuer"], f["title"], summary, doc, f["url"], f["at"][:10], []))
    return save_items(app, items)


READABLE = {"links.sgx.com": "https://www.sgx.com/", "nsearchives.nseindia.com": NSE_PAGE,
            "www.bseindia.com": "https://www.bseindia.com/", "cdn-api.markitdigital.com": "https://www.asx.com.au/"}


def read(url, chars=6000):
    """The text of one filing, for an agent that needs the detail behind a headline: an SGX announcement's page and
    PDF, or the PDF an NSE, BSE or ASX item links to. '' when the document is a scan or cannot be fetched."""
    host = urlsplit(url).hostname or ""
    if host not in READABLE:
        raise ValueError("not an exchange filing")
    s = _session(READABLE[host])
    text = " ".join(sgx_read(s, url)) if host == "links.sgx.com" else pdf_text(s, url, pages=6)
    return " ".join(text.split())[:chars]


# ---------- ASX: headlines only, so likely headlines of companies worth A$100M or more get their PDF read ----------
ASX = "https://asx.api.markitdigital.com/asx-research/1.0/"
ASX_FILE = "https://cdn-api.markitdigital.com/apiman-gateway/ASX/asx-research/1.0/file/"
ASX_MIN_CAP = 100_000_000
ASX_GEO = r"singapore|asia|asean|international|overseas|offshore|global|india"
ASX_ACTIONS = (r"subsidiar|incorporat|acqui|joint venture|memorandum|\bmou\b|establish|expansion|expand|partnership"
               r"|collaboration|new office|data cent(?:re|er)|facility|launch|enters|entry into")
ASX_NOISE = (r"appendix (?:2a|3|4)|change of director|(?:becoming a|ceasing to be a|change in) substantial"
             r"|notification regarding unquoted|notification of cessation|application for quotation"
             r"|cleansing notice|daily share buy-back|dividend|distribution|net tangible asset"
             r"|trading halt|pause in trading|quarterly|annual report|half year|agm|notice of|proxy form"
             r"|investor presentation|webinar")


def asx_candidate(headline, industry):
    """Read the PDF? Explorers (materials, energy) only when the headline names a geography."""
    h = headline.lower()
    if re.search(ASX_NOISE, h):
        return False
    return bool(re.search(ASX_GEO, h) or (industry not in ("Materials", "Energy") and re.search(ASX_ACTIONS, h)))


def asx(start, end):
    s = _session("https://www.asx.com.au/")
    listed = {}
    for r in csv.DictReader(io.StringIO(_get(s, ASX + "companies/directory/file", timeout=120).text)):
        cap = (r.get("Market Cap") or "").strip()
        if cap.replace(".", "").isdigit() and float(cap) >= ASX_MIN_CAP:
            listed[(r.get("ASX code") or "").strip().upper()] = (r.get("GICs industry group") or "").strip()
    for page in range(10):  # the feed is newest first and stops at 10,000 announcements, about 20 days
        rows = _get(s, ASX + "markets/announcements", params={"page": page, "itemsPerPage": 1000}).json()["data"]["items"]
        for r in rows:
            day, code, headline = (r.get("date") or "")[:10], (r.get("symbol") or "").upper(), r.get("headline") or ""
            if not (f"{start}" <= day <= f"{end}" and code in listed and r.get("documentKey")
                    and asx_candidate(headline, listed[code])):
                continue
            company = ((r.get("companyInfo") or [{}])[0].get("displayName") or code).title()
            doc = pdf_text(s, ASX_FILE + r["documentKey"])
            terms = hits("asx", f"{headline} {doc}"[:4000], company)
            if terms:
                yield item("asx", "AU", company, headline, "", doc, ASX_FILE + r["documentKey"], day, terms)
        if not rows or (rows[-1].get("date") or "")[:10] < f"{start}":
            break


EXCHANGES = {"nse": nse, "bse": bse, "sgx": sgx, "asx": asx}  # NSE before BSE: BSE repeats NSE-listed companies


def collect(con, days=14, today=None):
    """New filings from each exchange since its last good run (first run: `days` back, never more than 30), saved as
    items. An exchange that fails is skipped and picked up from the same day on the next run."""
    end, done, total = today or date.today(), db.kv_get(con, "filings_done", {}), 0
    for name, fetch in EXCHANGES.items():
        start = max(date.fromisoformat(done[name]) - timedelta(days=1) if name in done else end - timedelta(days=days),
                    end - timedelta(days=30))
        try:
            found = list(fetch(start, end))
        except Exception as e:  # a blocked or changed exchange must not stop the daily run
            print(f"  skip {name}: {type(e).__name__}: {e}")
            continue
        if name == "bse":  # the same disclosure filed at both Indian exchanges counts once: NSE's copy is kept
            on_nse = {(db.key(m[1]), r["published"]) for r in con.execute(
                "SELECT title, published FROM items WHERE source='nse'") if (m := COMPANY.search(r["title"]))}
            found = [i for i in found if (db.key(i["company"]), i["published"]) not in on_nse]
        total += save_items(con, found)
        done[name] = end.isoformat()
        db.kv_set(con, "filings_done", done)
        con.commit()
    return total


if __name__ == "__main__":
    assert hits("nse", "Updates. Incorporation of a wholly owned subsidiary in Singapore")
    assert not hits("nse", "Analyst meet in Singapore on the acquisition")
    assert not hits("nse", "Acme Investments Limited: board to meet in Singapore", "Acme Investments Limited")
    assert hits("sgx", "Incorporation of subsidiary. A new plant in Chennai") and hits("sgx", "Acquisition of Foo Pty Ltd")
    assert not hits("sgx", "The amount was paid in Sydney")  # "amount" is not an MoU
    assert asx_candidate("Expansion into Singapore", "Materials") and not asx_candidate("New facility", "Energy")
    assert excerpt("x" * 500 + " Singapore unit", ["singapore"], 90).startswith("x" * 29 + " Singapore")
    print("filings ok")
