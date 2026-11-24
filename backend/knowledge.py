"""Durable public knowledge beside the read-only registry graph.

Network collection returns evidence; the revenue worker alone commits it after
checking its lease. Model interpretation and contact data never enter this store.
"""
import hashlib
import http.client
import ipaddress
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import socket
import ssl
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote, quote_plus, unquote, urljoin, urlsplit, urlunsplit

from . import ctx, db, graph

SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_entities(
 uen TEXT PRIMARY KEY, company_id BIGINT UNIQUE, name TEXT NOT NULL, profile TEXT NOT NULL, updated TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS knowledge_monitors(
 uen TEXT PRIMARY KEY, history_collected BIGINT DEFAULT 0, checked_at TEXT, refresh_at TEXT,
 token TEXT, lease_until TEXT, error TEXT, warnings TEXT DEFAULT '[]');
CREATE TABLE IF NOT EXISTS knowledge_subscriptions(
 mission_id BIGINT, uen TEXT, PRIMARY KEY(mission_id,uen));
CREATE TABLE IF NOT EXISTS knowledge_documents(
 id BIGSERIAL PRIMARY KEY, uen TEXT NOT NULL, kind TEXT NOT NULL, url TEXT NOT NULL,
 title TEXT, text TEXT NOT NULL, published TEXT, fetched TEXT NOT NULL, hash TEXT NOT NULL,
 UNIQUE(uen,kind,url,hash));
CREATE INDEX IF NOT EXISTS knowledge_document_entity ON knowledge_documents(uen,fetched DESC,id DESC);
CREATE INDEX IF NOT EXISTS knowledge_monitor_due ON knowledge_monitors(refresh_at);
ALTER TABLE knowledge_monitors ADD COLUMN IF NOT EXISTS website_domain TEXT;
"""
MAX_BYTES = 2_000_000
MAX_PDF_BYTES = 12_000_000
PDF_PAGES = 20
ACRA = 'https://data.gov.sg/collections/2/view'
GEBIZ = 'https://www.gebiz.gov.sg/'


def stamp(at=None):
    return (at or datetime.now(timezone.utc)).isoformat(timespec='seconds')


def _address(url):
    p = urlsplit(url)
    if p.scheme not in ('https', 'http') or not p.hostname or p.username or p.password or p.port not in (None, 80, 443):
        raise ValueError('Only public HTTP/HTTPS source URLs are allowed')
    addresses = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == 'https' else 80), type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError('Source URL resolves to a private or reserved address')
    return p, addresses[0][4][0]


def fetch(url, allow_pdf=False):
    """Pin DNS to a validated public IP; retain original TLS SNI/Host. Validate each redirect before connecting."""
    for _ in range(4):
        p, address = _address(url)
        path=quote(p.path or '/',safe="/%:@!$&'()*+,;=-._~")
        query=quote(p.query,safe="%=&/:?@!$'()*+,;~-._")
        url=urlunsplit((p.scheme,p.netloc,path,query,''))
        port = p.port or (443 if p.scheme == 'https' else 80)
        con = http.client.HTTPConnection(p.hostname, port, timeout=20)
        try:
            raw = socket.create_connection((address, port), timeout=20)
            try:
                con.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=p.hostname) if p.scheme == 'https' else raw
            except Exception:
                raw.close()
                raise
            con.request('GET', path + ('?' + query if query else ''), headers={
                'User-Agent': 'SignalResearch/1.0 (enterprise public-source research)', 'Accept-Encoding': 'identity'})
            response = con.getresponse()
            if response.status in (301,302,303,307,308):
                location = response.getheader('Location')
                if not location:
                    raise ValueError('Source redirect has no destination')
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f'Source returned HTTP {response.status}')
            limit = MAX_PDF_BYTES if allow_pdf else MAX_BYTES
            if int(response.getheader('Content-Length') or 0) > limit:
                raise ValueError('Source exceeds collection size limit')
            content = response.read(limit + 1)
            if len(content) > limit:
                raise ValueError('Source exceeds collection size limit')
            kind = (response.getheader('Content-Type') or '').lower()
            if allow_pdf and content.startswith(b'%PDF-'):
                return {'url':url,'text':'','content':content,'type':'application/pdf'}
            if not any(k in kind for k in ('text/', 'xml', 'json')):
                raise ValueError('Source is not readable HTML, text, XML or JSON')
            return {'url': url, 'text': content.decode('utf-8', errors='replace'), 'type': kind}
        finally:
            con.close()
    raise ValueError('Source redirected too many times')


def extract_pdf(content):
    """Bound annual-report parsing by bytes, pages, CPU, memory, output size and elapsed time."""
    tool, limits = shutil.which('pdftotext'), shutil.which('prlimit')
    if not tool or not limits:
        raise ValueError('Annual-report extraction needs pdftotext and prlimit on the worker')
    if len(content)>MAX_PDF_BYTES or not content.startswith(b'%PDF-'):
        raise ValueError('Annual-report PDF is invalid or exceeds the 12 MB limit')
    with tempfile.TemporaryDirectory(prefix='signal-report-') as directory:
        source,target=Path(directory)/'report.pdf',Path(directory)/'report.txt'
        source.write_bytes(content)
        try:
            result=subprocess.run([limits,'--as=536870912','--cpu=20','--fsize=2097152','--',tool,'-f','1','-l',str(PDF_PAGES),'-enc','UTF-8',str(source),str(target)],
                                  stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
        except subprocess.TimeoutExpired as e:
            raise ValueError('Annual-report extraction exceeded its time limit') from e
        if result.returncode or not target.exists():
            raise ValueError('Annual-report PDF text could not be extracted; encrypted/scanned reports need review')
        pages=target.read_text(errors='replace').split('\f')[:PDF_PAGES]
        if not any(p.strip() for p in pages):
            raise ValueError('Annual report has no extractable text; OCR needs review')
        return [(i+1,'\n'.join(pages[i:i+5]).strip()[:14000]) for i in range(0,len(pages),5) if any(p.strip() for p in pages[i:i+5])]


def _plain(raw):
    page = ctx._Page()
    page.feed(raw)
    return '\n'.join(page.text)[:14000], page.links


def _serial(value):
    return json.loads(json.dumps(value, default=lambda r: dict(r)))


def collect(g, uen, company, history=True):
    """Registry history + real news backfill/current feed + available verified company pages. No database writes."""
    p = graph.profile(g, uen)
    if not p:
        raise ValueError('Canonical UEN is absent from the registry; identity cannot be verified')
    p = _serial(p)
    name, docs, warnings = p['e']['name'], [], []
    aliases = {db.key(name), db.key(company.get('name') or name), *(db.key(x) for x in p['keys'])}
    aliases.discard('')
    aliases |= {alias for alias,canonical in db.ALIASES.items() if canonical in aliases}
    def named(text):
        value = db.key(text, alias=False)
        return uen.lower() in text.lower() or any(len(k)>=4 and (' '+k+' ') in (' '+value+' ') for k in aliases)
    def add(kind, url, title, text, published=None):
        if text.strip():
            text=re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}','[address omitted]',text)
            docs.append({'kind':kind,'url':url,'title':title,'text':text[:14000],'published':published})
    add('registry', ACRA, 'ACRA registry identity', f"{name}; UEN {uen}; {p['e']['status']}; registered {p['e']['registered']}. Activity: {p['e']['ssic_desc'] or p['e']['activity'] or 'not stated'}.")
    for parent in p['parents']:
        add('ownership', f"https://search.gleif.org/#/record/{parent['lei']}", 'GLEIF ownership', f"{name} is linked to parent {parent['name']} ({parent['country']}); {parent['kind']}.")
    for c in p['won'][:20]:
        add('contract', GEBIZ, c['description'], f"{name} supplied {c['description']} to {c['agency']}; awarded {c['awarded']}; value SGD {c['amount']}. This is a supplier award, not evidence of {name}'s own telecom provider or renewal.", c['awarded'])
    for n in p['news'][:20]:
        add('news', n['url'], n['title'] or 'Graph news', n['title'] or n['url'], (n['at'] or '')[:10] or None)
    query = ' OR '.join('"'+x+'"' for x in dict.fromkeys([company.get('name') or name, name]))
    rss = 'https://news.google.com/rss/search?q='+quote_plus(query + (' after:'+(datetime.now(timezone.utc)-timedelta(days=365)).date().isoformat() if history else ' when:7d'))+'&hl=en-SG&gl=SG&ceid=SG:en'
    news_urls = [n['url'] for n in p['news'][:2]]
    history_complete = True
    try:
        feed = fetch(rss)
        for item in list(ET.fromstring(feed['text']).iter('item'))[:30]:
            title, url = item.findtext('title') or '', item.findtext('link') or ''
            if not named(title) or not url.startswith('https://news.google.com/'):
                continue
            published = item.findtext('pubDate')
            d = parsedate_to_datetime(published).date().isoformat() if published else None
            add('news', url, title, title+'\n'+_plain(item.findtext('description') or '')[0], d)
            if len(news_urls)<2:
                news_urls.append(url)
    except (ValueError, OSError, ET.ParseError, http.client.HTTPException):
        history_complete = False
        warnings.append('News backfill/refresh unavailable; saved registry history remains available.')
    domain = (company.get('domain') or '').strip().lower()
    reports=[]
    def links_from(base, links):
        for link in links:
            u=urljoin(base,link).split('#')[0]
            name=unquote(u).lower()
            annual='annual' in name or bool(re.search(r'(?:[/_\s-]|^)ar[\s_-]?\d{2,4}',name))
            if u.lower().split('?',1)[0].endswith('.pdf') and annual:
                if u not in reports:reports.append(u)
    if domain:
        try:
            home = fetch('https://'+domain)
            text, links = _plain(home['text'])
            if not named(text):
                raise ValueError('Company page does not identify the registry entity')
            add('website', home['url'], f'{name} company website', text)
            links_from(home['url'],links)
            pick = []
            for link in links:
                u = urljoin(home['url'], link).split('#')[0]
                if urlsplit(u).hostname == urlsplit(home['url']).hostname and any(w in u.lower() for w in ('investor','annual','news','about')) and u not in pick and not u.lower().endswith('.pdf'):
                    pick.append(u)
            pick.sort(key=lambda u:next((i for i,k in enumerate(('annual','investor','financial','news','about')) if k in u.lower()),9))
            visited={home['url']}
            checked=0
            while pick and checked<4:
                u=pick.pop(0)
                if u in visited:continue
                visited.add(u);checked+=1
                try:
                    page = fetch(u,allow_pdf=True)
                    if page['type']=='application/pdf':
                        if page['url'] not in reports:reports.append(page['url'])
                        continue
                    text, page_links = _plain(page['text'])
                    if named(text):
                        add('website', page['url'], f'{name} company information', text)
                        links_from(page['url'],page_links)
                        if not reports:
                            for link in page_links:
                                target=urljoin(page['url'],link).split('#')[0]
                                if 'annual' in target.lower() and not target.lower().endswith('.pdf') and urlsplit(target).hostname==urlsplit(home['url']).hostname and target not in visited and target not in pick:
                                    pick.insert(0,target)
                except (ValueError, OSError, http.client.HTTPException):
                    warnings.append('A company information page could not be read.')
        except (ValueError, OSError, http.client.HTTPException):
            warnings.append('Verified company website unavailable; no guessed domain was used.')
    else:
        warnings.append('No verified company website saved; no domain was guessed.')
    if reports:
        reports.sort(key=lambda u:max([int(y) for y in re.findall(r'20\d\d',u)] or [0]),reverse=True)
        try:
            report=fetch(reports[0],allow_pdf=True)
            chunks=extract_pdf(report.get('content',b''))
            if not named(' '.join(text for _,text in chunks)):
                raise ValueError('Annual report does not identify this company')
            for page,text in chunks:
                add('annual_report',report['url'].split('#')[0]+f'#page={page}',f'{name} annual report · pages {page}–{min(page+4,PDF_PAGES)} (bounded excerpt)',text)
            warnings.append(f'Annual report coverage: first {PDF_PAGES} pages only, with bounded text excerpts. Later pages were not read.')
        except (ValueError,OSError,http.client.HTTPException):
            warnings.append('Annual report was found but PDF extraction was unavailable or failed; no report facts were invented.')
    elif domain:
        warnings.append('No annual-report PDF found in the bounded company-site crawl; report coverage remains unknown.')
    for u in news_urls[:2]:
        try:
            page = fetch(u)
            text, _ = _plain(page['text'])
            if named(text) and len(text)>120:
                add('news', page['url'], f'{name} public coverage', text)
        except (ValueError, OSError, http.client.HTTPException):
            warnings.append('An article body was unavailable; its sourced headline is retained.')
    return {'profile':p,'documents':docs,'warnings':list(dict.fromkeys(warnings)), 'history_complete':history_complete}


def bind(con, uen, profile, at):
    """Canonical UEN wins over a colliding normalized name. Caller owns transaction and fencing."""
    existing = con.execute('SELECT company_id FROM knowledge_entities WHERE uen=?', (uen,)).fetchone()
    if existing:
        cid = existing[0]
    else:
        e = profile['e']
        same = con.execute('SELECT id FROM companies WHERE uen=? ORDER BY in_crm DESC,id LIMIT 1', (uen,)).fetchone()
        if same:
            cid = same[0]
        else:
            key = db.key(e['name'])
            collision = con.execute('SELECT id,uen FROM companies WHERE key=?', (key,)).fetchone()
            # Never identify a registry entity by fuzzy company name alone.
            if collision:
                key = 'uen:'+uen.lower()
            cid = con.execute('INSERT INTO companies(key,name,uen,hq_country,industry,watch) VALUES(?,?,?,\'SG\',?,\'Revenue research\') ON CONFLICT(key) DO NOTHING',
                              (key,e['name'],uen,(e.get('ssic_desc') or 'other').lower())).lastrowid
            if not cid:
                cid = con.execute('SELECT id FROM companies WHERE key=? AND uen=?', (key,uen)).fetchone()[0]
    con.execute('INSERT INTO knowledge_entities(uen,company_id,name,profile,updated) VALUES(?,?,?,?,?) ON CONFLICT(uen) DO UPDATE SET profile=excluded.profile,updated=excluded.updated',
                (uen,cid,profile['e']['name'],json.dumps(profile,default=lambda r:dict(r)),stamp(at)))
    return cid


def save(con, uen, bundle, at):
    cid = bind(con, uen, bundle['profile'], at)
    for d in bundle['documents']:
        digest = hashlib.sha256(d['text'].encode()).hexdigest()
        con.execute('INSERT INTO knowledge_documents(uen,kind,url,title,text,published,fetched,hash) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING',
                    (uen,d['kind'],d['url'],d['title'],d['text'],d.get('published'),stamp(at),digest))
    return cid


def documents(con, uen, limit=60):
    return [dict(r) for r in con.execute('SELECT * FROM (SELECT DISTINCT ON(kind,url,title) * FROM knowledge_documents WHERE uen=? ORDER BY kind,url,title,fetched DESC,id DESC) d ORDER BY fetched DESC,id DESC LIMIT ?', (uen,limit))]


def base_profile(con, uen):
    row = con.execute('SELECT profile FROM knowledge_entities WHERE uen=?', (uen,)).fetchone()
    return json.loads(row[0]) if row else None


def overlay(con, uen):
    docs = documents(con, uen)
    return [{'id':f"runtime:{d['id']}",'title':d['title'],'url':d['url'],'at':d['published'] or d['fetched'],
             'source':'Collected public source','lang':'eng','themes':'','orgs':'','persons':''} for d in docs if d['kind']=='news'], docs
