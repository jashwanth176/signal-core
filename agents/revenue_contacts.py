"""Free contact discovery and narrowly approved paid reveal for a revenue child.

A grant covers one company/person and at most one demographic/work-email credit.
Attempts are durable before dispatch; an ambiguous request is never retried.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from backend import db, people, privacy
from . import outreach

SCHEMA = """
CREATE TABLE IF NOT EXISTS revenue_contact_grants(
 id BIGSERIAL PRIMARY KEY, child_id BIGINT NOT NULL, mission_id BIGINT NOT NULL, owner TEXT NOT NULL,
 person_id BIGINT NOT NULL, max_credits BIGINT DEFAULT 1, remaining BIGINT DEFAULT 1,
 expires TEXT NOT NULL, state TEXT DEFAULT 'active', created TEXT, UNIQUE(child_id,person_id));
CREATE TABLE IF NOT EXISTS revenue_reveal_attempts(
 id BIGSERIAL PRIMARY KEY, grant_id BIGINT UNIQUE, child_id BIGINT, person_id BIGINT, token TEXT,
 state TEXT DEFAULT 'reserved', created TEXT, finished TEXT, note TEXT);
"""
SOURCE = 'https://docs.apollo.io/reference/people-api-search'


class NeedsApproval(RuntimeError):
    pass


def _suppressed(con, p, domain):
    return bool(privacy.forgotten(con,[p['ext_id']])) or bool(domain and outreach.stopped(con,'contact@'+domain)) or bool(p['email'] and outreach.stopped(con,p['email']))


def discover(g, company):
    """Only the documented zero-credit endpoint. Company search/enrichment are not a fallback."""
    if not os.environ.get('APOLLO_API_KEY'):
        return {'people':[],'status':'needs_contact','note':'Contact discovery needs an Apollo search key or sourced public contacts; no contacts were invented.'}
    domain = (company.get('domain') or '').lower()
    filters = {'per_page':25,'page':1,'person_seniorities':['c_suite','vp','head','director']}
    if domain:
        filters['q_organization_domains_list']=[domain]
    else:
        filters.update(q_keywords=company['name'],organization_locations=['Singapore'])
    rows = people._post('/mixed_people/api_search',filters).get('people',[])
    if not rows and not domain and db.key(company['name'])!=company['name'].lower():
        rows = people._post('/mixed_people/api_search',filters|{'q_keywords':db.key(company['name'])}).get('people',[])
    found=[]
    for p in rows:
        org=p.get('organization') or {}
        if not p.get('id') or not people.TECH.search(p.get('title') or '') or people.SKIP.search(p.get('title') or ''):
            continue
        # A broad keyword response is never accepted without an exact organization match.
        if db.key(org.get('name') or '')!=db.key(company['name']):
            continue
        org_domain=(org.get('primary_domain') or urlsplit(org.get('website_url') or '').hostname or '').lower().removeprefix('www.')
        if domain and org_domain and org_domain!=domain.removeprefix('www.'):
            continue
        count=g.execute("SELECT count(*) FROM entities WHERE key=? AND status LIKE 'Live%'",(db.key(company['name']),)).fetchone()[0]
        if not domain and count!=1:
            continue  # colliding legal entities need a verified domain, not a guessed binding
        found.append({'id':p['id'],'first_name':p.get('first_name'),'last_name_obfuscated':p.get('last_name_obfuscated') or '',
                      'title':p.get('title'),'organization_id':org.get('id'),'organization_name':org.get('name'),
                      'domain':domain or org_domain,'source':SOURCE})
    return {'people':found[:10],'status':'needs_approval' if found else 'needs_contact',
            'note':'Discovered sourced people; a separate one-person credit grant is required for paid work-email reveal.' if found else 'No identity-matched senior contacts found in the permitted free search.'}


def save_discovered(con, company, result, at):
    domain=company.get('domain')
    for p in sorted(result['people'],key=lambda p:p['id']):
        privacy.identity_lock(con,p['id'])
        if privacy.forgotten(con,[p['id']]):
            continue
        existing=con.execute('SELECT id,company_id FROM people WHERE ext_id=?',(p['id'],)).fetchone()
        if existing:
            continue  # never move a global Apollo identity or replace a revealed name with a masked preview
        name=' '.join(x for x in (p.get('first_name'),p.get('last_name_obfuscated')) if x)
        if not name:
            continue
        con.execute("INSERT INTO people(company_id,ext_id,name,title,source,updated,source_url,source_at,source_meta) VALUES(?,?,?,?,'apollo',?,?,?,?) ON CONFLICT(ext_id) DO NOTHING",
                    (company['id'],p['id'],name,p['title'],at.isoformat(),SOURCE,at.isoformat(),json.dumps({'uen':company['uen'],'organization_id':p['organization_id'],'organization_name':p['organization_name'],'domain':p['domain'],'match':'exact organization name / registry identity'})))
        if not domain and p.get('domain'):
            domain=p['domain']
    if domain and not company.get('domain'):
        con.execute("UPDATE companies SET domain=? WHERE id=? AND (domain IS NULL OR domain='')",(domain,company['id']))


def grant(con, child_id, owner, person_id, expires):
    from . import revenue
    at=revenue.now()
    end=datetime.fromisoformat(expires)
    if not end.tzinfo or not at<end<=at+timedelta(hours=24,seconds=2):
        raise ValueError('Approve an explicit expiry within the next 24 hours')
    end=end.astimezone(timezone.utc)
    with con.transaction():
        m=con.execute('SELECT m.* FROM revenue_missions m JOIN revenue_children c ON c.mission_id=m.id WHERE c.id=? AND m.owner=? FOR UPDATE OF m',(child_id,owner)).fetchone()
        if not m or m['state']=='stopped':
            raise ValueError('This mission is stopped or not yours')
        until=json.loads(m['config'])['schedule']['until']
        if end.astimezone(revenue.SGT).date().isoformat()>until:
            raise ValueError('The grant cannot outlive the approved mission')
        child=con.execute('SELECT * FROM revenue_children WHERE id=? FOR UPDATE',(child_id,)).fetchone()
        p=con.execute('SELECT * FROM people WHERE id=? AND company_id=?',(person_id,child['company_id'])).fetchone()
        co=con.execute('SELECT domain FROM companies WHERE id=?',(child['company_id'],)).fetchone()
        if not p or p['source']!='apollo' or not p['ext_id'] or _suppressed(con,p,co['domain']):
            raise ValueError('A current, unsuppressed Apollo person at this company is required')
        gid=con.execute('INSERT INTO revenue_contact_grants(child_id,mission_id,owner,person_id,expires,created) VALUES(?,?,?,?,?,?) ON CONFLICT DO NOTHING',(child_id,m['id'],owner,person_id,end.isoformat(),at.isoformat())).lastrowid
        if not gid:
            return con.execute('SELECT id FROM revenue_contact_grants WHERE child_id=? AND person_id=?',(child_id,person_id)).fetchone()[0]
        con.execute("UPDATE revenue_children SET state='queued',stage='people',attempts=0,next_at=?,error=NULL WHERE id=?",(revenue.stamp(at),child_id))
        con.execute("UPDATE revenue_missions SET state=CASE WHEN state='completed' THEN 'finishing' ELSE state END WHERE id=?",(m['id'],))
        revenue.event(con,m['id'],child['run_id'],child_id,'contact approval','Seller approved one work-email reveal credit for one sourced person, with explicit expiry.',at)
        return gid


def reveal(con, job, company, at):
    from . import revenue
    at=revenue.now()
    with con.transaction():
        revenue._fence(con,job,at)
        grant=con.execute("SELECT * FROM revenue_contact_grants WHERE child_id=? AND state='active' ORDER BY id DESC LIMIT 1 FOR UPDATE",(job['id'],)).fetchone()
        if not grant:
            return None
        existing=con.execute('SELECT state FROM revenue_reveal_attempts WHERE grant_id=?',(grant['id'],)).fetchone()
        if existing:
            return {'status':'needs_approval','note':'A reveal was already attempted. No automatic retry or extra credits; reconcile its outcome before new authorization.'}
        p=con.execute('SELECT * FROM people WHERE id=? AND company_id=?',(grant['person_id'],company['id'])).fetchone()
        if datetime.fromisoformat(grant['expires'])<=at or grant['remaining']<1 or grant['owner']!=revenue.get(con,job['mission_id'])['owner']:
            return {'status':'needs_approval','note':'The one-person reveal grant is expired or exhausted.'}
        if not p or _suppressed(con,p,company.get('domain')):
            return {'status':'needs_contact','note':'The selected person is no longer permitted at this company.'}
        if not os.environ.get('APOLLO_API_KEY'):
            return {'status':'needs_approval','note':'An Apollo key is required before the approved reveal can run.'}
        if people.spend(con,1)!=1:
            return {'status':'needs_approval','note':'The global Apollo daily cap is exhausted; this grant cannot spend past it.'}
        aid=con.execute('INSERT INTO revenue_reveal_attempts(grant_id,child_id,person_id,token,created) VALUES(?,?,?,?,?)',(grant['id'],job['id'],p['id'],job['token'],at.isoformat())).lastrowid
        con.execute('UPDATE revenue_contact_grants SET remaining=remaining-1 WHERE id=?',(grant['id'],))
    # Persist dispatch before network; no exception or crash can silently replay the same grant.
    with con.transaction():
        dispatch_at=revenue.now()
        revenue._fence(con,job,dispatch_at)
        live_grant=con.execute('SELECT * FROM revenue_contact_grants WHERE id=? FOR UPDATE',(grant['id'],)).fetchone()
        privacy.identity_lock(con,p['ext_id'])
        current=con.execute('SELECT * FROM people WHERE id=? AND company_id=?',(p['id'],company['id'])).fetchone()
        # Locks above can wait beyond the consent or lease deadline.
        dispatch_at=revenue.now()
        revenue._fence(con,job,dispatch_at)
        if not live_grant or live_grant['state']!='active' or datetime.fromisoformat(live_grant['expires'])<=dispatch_at:
            con.execute("UPDATE revenue_reveal_attempts SET state='cancelled',note='Grant expired or was revoked before dispatch; no paid request made',finished=? WHERE id=?",(dispatch_at.isoformat(),aid))
            return {'status':'needs_approval','note':'The grant expired before dispatch. No paid request was made; no automatic repeat is allowed.'}
        if not current or _suppressed(con,current,company.get('domain')):
            con.execute("UPDATE revenue_reveal_attempts SET state='cancelled',note='Person removed or suppressed before dispatch; no paid request made',finished=? WHERE id=?",(revenue.stamp(),aid))
            return {'status':'needs_contact','note':'The selected person was removed before dispatch. No paid request was made.'}
        con.execute("UPDATE revenue_reveal_attempts SET state='started' WHERE id=? AND state='reserved'",(aid,))
    try:
        matches=people.enrich([p['ext_id']])
    except Exception:
        con.execute("UPDATE revenue_reveal_attempts SET state='uncertain',note='Request outcome unknown; reservation retained; no automatic retry',finished=? WHERE id=?",(revenue.stamp(),aid))
        return {'status':'needs_approval','note':'The paid request outcome is uncertain. Credit reservation is retained; it will not automatically retry.'}
    return {'status':'checked','attempt_id':aid,'person_id':p['id'],'matches':matches}


def save_reveal(con, job, result, company, at):
    p=con.execute('SELECT * FROM people WHERE id=? AND company_id=?',(result['person_id'],company['id'])).fetchone()
    if p:
        privacy.identity_lock(con,p['ext_id'])
        p=con.execute('SELECT * FROM people WHERE id=? AND company_id=?',(result['person_id'],company['id'])).fetchone()
    if not p or _suppressed(con,p,company.get('domain')):
        raise ValueError('Person was removed or suppressed before result storage')
    match=next((m for m in result['matches'] if m.get('id')==p['ext_id']),None)
    usable=False
    if match:
        org=match.get('organization') or {}
        domain=(org.get('primary_domain') or urlsplit(org.get('website_url') or '').hostname or company.get('domain') or '').lower()
        identity=db.key(org.get('name') or '')==db.key(company['name'])
        expected=company.get('domain')
        if identity and (not expected or domain.removeprefix('www.')==expected.removeprefix('www.')):
            email=people.work_email(match.get('email'),domain)
            status=match.get('email_status')
            usable=bool(email and status=='verified' and not outreach.stopped(con,email))
            history=match.get('employment_history')
            alumni,started=people.flags(history or [])
            con.execute('UPDATE people SET name=coalesce(?,name),title=coalesce(?,title),email=?,email_status=?,history=coalesce(?,history),alumni=coalesce(?,alumni),started=coalesce(?,started),updated=?,source_at=?,source_url=?,source_meta=? WHERE id=? AND company_id=?',
                        (match.get('name'),match.get('title'),email,status,people._hist(history) if history is not None else None,alumni,started,at.isoformat(),at.isoformat(),'https://docs.apollo.io/reference/bulk-people-enrichment',
                         json.dumps({'uen':company['uen'],'organization_name':org.get('name'),'attempt_id':result['attempt_id'],'scope':'work-email only'}),p['id'],company['id']))
            if domain and not expected:
                con.execute('UPDATE companies SET domain=? WHERE id=?',(domain,company['id']))
    con.execute("UPDATE revenue_reveal_attempts SET state=?,note=?,finished=? WHERE id=? AND token=?",('succeeded' if usable else 'no_verified_match','Stored only an identity-matched work contact; no automatic repeat',at.isoformat(),result['attempt_id'],job['token']))
    return usable
