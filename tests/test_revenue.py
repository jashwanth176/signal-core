"""Durable revenue work: graph identities, actual saved stages and explicit seller actions."""
import json
import os
from datetime import datetime, timedelta, timezone
from unittest import mock

from fastapi.testclient import TestClient

from agents import llm, thread
from backend import db, kg
from frontend import web
from tests.helpers import Base, seed_graph

SP = '199201623M'
NINJA = '201000002B'


class RevenueBase(Base):
    def setUp(self):
        super().setUp()
        seed_graph(self.g)
        kg.build(self.g)
        kg.link(self.con, self.g)
        self.at = datetime.now(timezone.utc)

    def mission(self, **kw):
        from agents import revenue
        until = (self.at.astimezone(revenue.SGT).date() + timedelta(days=6)).isoformat()
        cfg = {'time': '09:00', 'until': until, 'limit': 1, 'uens': [SP, NINJA], 'include_customers': True, **kw}
        tid = thread.new(self.con, 'Aisha Tan', 'Research fresh companies each day')
        out = revenue.propose(self.con, 'Aisha Tan', tid, 'Research fresh companies and draft sales emails', **cfg)
        thread.decide(self.con, out['waiting'], 'approve')
        return self.con.execute('SELECT id FROM revenue_missions ORDER BY id DESC LIMIT 1').fetchone()[0]

    def prepare(self):
        from backend import people
        self.mission(uens=[NINJA])
        self.preview={'id':'discover-1','first_name':'Ana','last_name_obfuscated':'T***','title':'Chief Technology Officer',
                      'organization':{'id':'org-1','name':'NINJA LOGISTICS PTE. LTD.','primary_domain':'ninjavan.co'}}
        with mock.patch.dict(os.environ,{'APOLLO_API_KEY':'test'}),mock.patch.object(people,'_post',return_value={'people':[self.preview]}):
            self.drain()
        return dict(self.con.execute('SELECT * FROM revenue_children').fetchone()),dict(self.con.execute("SELECT * FROM people WHERE ext_id='discover-1'").fetchone())

    def fake_sources(self, g, uen, company, history=True):
        from backend import graph
        p = json.loads(json.dumps(graph.profile(g, uen),default=lambda r:dict(r)))
        docs = [{'kind': 'news', 'url': 'https://news.example/' + uen,
                 'title': 'Expansion', 'published': '2026-11-10', 'text': f'{p["e"]["name"]} opens a distribution hub in Singapore.'}]
        if company.get('domain'):
            docs.append({'kind':'website','url':'https://'+company['domain'],'title':'Company website','text':p['e']['name']+' operates a distribution network.','published':None})
        return {'profile': p, 'documents': docs, 'warnings': []}

    def fake_model(self, schema, system, user, **kw):
        from agents import revenue
        ctx = json.loads(user)
        if schema is revenue.Research:
            d = next(d for d in ctx['documents'] if d['kind'] == 'news')
            claim = revenue.Claim(text=d['text'], source_id=d['id'], quote=d['text'], kind='fact')
            return revenue.Research(why_now=[claim], renewals=[], competitors=[], approach='Discuss network reliability for the new hub.',
                                    unknowns=['Incumbent and contract end are not verified.'],profile_sections=[revenue.ProfileSection(kind='company',title='Business and operations',findings=[claim])])
        return revenue.Email(subject='Your next hub', body='Hi,\nCould we discuss reliable connectivity for your expansion?\nAisha',
                             source_ids=[d['id'] for d in ctx['documents'] if d['kind'] == 'news'])

    def drain(self, count=10, at=None):
        from agents import revenue
        with mock.patch('backend.knowledge.collect', self.fake_sources), mock.patch.object(llm, 'live', return_value=True), mock.patch.object(llm, 'ask', self.fake_model):
            for _ in range(count):
                if revenue.work_once(self.con, self.g, at=at) is None:
                    break


class MissionTest(RevenueBase):
    def test_due_monitor_collects_again_and_keeps_original_citation(self):
        from agents import revenue
        self.mission()
        self.drain()
        child = self.con.execute('SELECT * FROM revenue_children').fetchone()
        old = json.loads(child['artifact'])['why_now'][0]['source_id']
        due = self.at + timedelta(days=1, minutes=1)
        def fresh(*args,**kw):
            b=self.fake_sources(*args,**kw)
            b['documents'][0]['text']+=' A second hub is planned.'
            return b
        with mock.patch('backend.knowledge.collect',fresh):
            revenue.refresh_monitor(self.con,self.g,at=due)
        view = revenue.child_view(self.con,child['id'],'Aisha Tan')
        self.assertIn(old,{d['id'] for d in view['documents']})
        self.assertGreater(self.count('knowledge_documents'),1)

    def test_competing_workers_claim_one_child(self):
        from agents import revenue
        from concurrent.futures import ThreadPoolExecutor
        self.mission()
        revenue.work_once(self.con,self.g)
        def take(_):
            with db.connect() as c:
                return revenue.claim(c)
        with ThreadPoolExecutor(2) as pool:
            claims=list(pool.map(take,range(2)))
        self.assertEqual(sum(x is not None for x in claims),1)

    def test_missing_model_is_visible_blocker_not_a_completed_draft(self):
        from agents import revenue
        self.mission()
        revenue.work_once(self.con,self.g)
        with mock.patch('backend.knowledge.collect',self.fake_sources):
            revenue.work_once(self.con,self.g)
        with mock.patch.object(llm,'live',return_value=False):
            for _ in range(2):
                revenue.work_once(self.con,self.g)
        row=self.con.execute('SELECT * FROM revenue_children').fetchone()
        self.assertEqual(row['state'],'blocked')
        self.assertIn('model key',row['error'])
        self.assertIsNone(row['email_body'])

    def test_new_company_discovers_person_then_requires_bounded_reveal_grant(self):
        from agents import revenue, revenue_contacts
        from backend import people
        self.mission(uens=[NINJA])
        found={'id':'new-person','first_name':'Ana','last_name_obfuscated':'T***','title':'Chief Technology Officer',
               'organization':{'id':'ninja-org','name':'NINJA LOGISTICS PTE. LTD.','primary_domain':'ninjavan.co'}}
        self.assertEqual(self.con.execute("SELECT count(*) FROM people WHERE ext_id='new-person'").fetchone()[0],0)
        with mock.patch.dict(os.environ,{'APOLLO_API_KEY':'free-search-test'}), mock.patch.object(people,'_post',return_value={'people':[found]}), mock.patch.object(people,'enrich',side_effect=AssertionError('No reveal without consent')):
            self.drain()
        child=self.con.execute('SELECT * FROM revenue_children').fetchone()
        self.assertEqual(child['state'],'needs_approval')
        self.assertTrue(child['email_body'])
        person=self.con.execute("SELECT * FROM people WHERE ext_id='new-person'").fetchone()
        self.assertEqual(person['source'],'apollo')
        self.assertIsNone(person['email'])
        expiry=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
        gid=revenue_contacts.grant(self.con,child['id'],'Aisha Tan',person['id'],expiry)
        self.assertEqual(revenue_contacts.grant(self.con,child['id'],'Aisha Tan',person['id'],expiry),gid)
        reveal=found|{'name':'Ana Tan','email':'ana@ninjavan.co','email_status':'verified'}
        with mock.patch.dict(os.environ,{'APOLLO_API_KEY':'test'}), mock.patch.object(people,'CREDIT_CAP',1), mock.patch.object(people,'enrich',return_value=[reveal]) as paid:
            self.drain()
        self.assertEqual(paid.call_count,1)  # adapter stub only; no paid network request in acceptance
        child=self.con.execute('SELECT * FROM revenue_children').fetchone()
        self.assertEqual(child['state'],'ready')
        self.assertEqual(self.con.execute('SELECT remaining FROM revenue_contact_grants WHERE id=?',(gid,)).fetchone()[0],0)
        self.assertEqual(self.con.execute('SELECT count(*) FROM revenue_reveal_attempts').fetchone()[0],1)

    def test_approval_enqueues_without_report_and_duplicate_is_safe(self):
        from agents import revenue
        tid = thread.new(self.con, 'Aisha Tan', 'New companies')
        until = (self.at.astimezone(revenue.SGT).date() + timedelta(days=6)).isoformat()
        out = revenue.propose(self.con, 'Aisha Tan', tid, 'Ten fresh companies', until=until, limit=10)
        self.assertEqual(self.count('revenue_missions'), 0)
        client = TestClient(web.app, follow_redirects=False)
        client.post(f"/ask/step/{out['waiting']}/approve")
        client.post(f"/ask/step/{out['waiting']}/approve")
        self.assertEqual(self.count('revenue_missions'), 1)
        self.assertEqual(self.count('revenue_runs'), 1)
        self.assertEqual(self.count('home_items'), 0)
        self.assertEqual(self.count('revenue_children'), 0)
        self.assertIn('queued', client.get(f'/ask/{tid}').text.lower())

    def test_worker_builds_monitor_overlay_research_and_draft(self):
        from agents import revenue
        from backend import graph, knowledge
        mid = self.mission()
        self.drain()
        child = self.con.execute('SELECT * FROM revenue_children').fetchone()
        self.assertEqual(child['state'], 'needs_contact')
        self.assertTrue(child['email_subject'] and child['email_body'])
        self.assertEqual(self.count('knowledge_monitors'), 1)
        self.assertTrue(knowledge.documents(self.con, SP))
        self.assertTrue(graph.profile(self.g, SP, app=self.con)['runtime'])
        self.assertTrue([e for e in kg.edges(self.g, self.con, SP, ['evidence']) if e['source'].startswith('https')])
        # No implicit paid contact lookups, customer messages or Salesforce records.
        self.assertEqual(self.count('tasks'), 0)
        self.assertIn('news.example', json.dumps(revenue.child_view(self.con, child['id'], 'Aisha Tan')))
        self.assertIsNone(revenue.child_view(self.con, child['id'], 'Someone Else'))

    def test_stale_claim_cannot_save_after_stop_or_reclaim(self):
        from agents import revenue
        mid = self.mission()
        revenue.work_once(self.con, self.g, self.at)
        claim = revenue.claim(self.con, self.at)
        self.assertIsNotNone(claim)
        revenue.stop(self.con, mid, 'Aisha Tan')
        with self.assertRaises(revenue.LostLease):
            revenue.complete(self.con, claim, {'bundle': self.fake_sources(self.g, SP, {})}, self.at)
        self.assertEqual(self.count('knowledge_documents'), 0)
        self.assertEqual(self.con.execute('SELECT state FROM revenue_children').fetchone()[0], 'cancelled')

    def test_leases_recover_and_old_worker_is_fenced(self):
        from agents import revenue
        self.mission()
        revenue.work_once(self.con, self.g)
        old = revenue.claim(self.con, self.at)
        later = self.at + timedelta(minutes=20)
        revenue.maintain(self.con, later)
        new = revenue.claim(self.con, later)
        self.assertNotEqual(new['token'], old['token'])
        with self.assertRaises(revenue.LostLease):
            revenue.complete(self.con, old, {}, later)
        self.assertEqual(self.con.execute('SELECT token FROM revenue_children').fetchone()[0], new['token'])

    def test_delivery_is_nonrepeating_and_missing_contacts_are_not_ready(self):
        from agents import revenue
        mid = self.mission()
        self.drain()
        run = self.con.execute('SELECT * FROM revenue_runs ORDER BY id').fetchone()
        due = datetime.fromisoformat(run['due_at'])
        revenue.maintain(self.con, due)
        self.assertEqual(self.count('revenue_deliveries'), 1)
        revenue.maintain(self.con, due)
        self.assertEqual(self.count('revenue_deliveries'), 1)
        self.drain(at=due)
        children = self.con.execute('SELECT uen FROM revenue_children ORDER BY id').fetchall()
        self.assertEqual(len({c['uen'] for c in children}), len(children))
        item = self.con.execute('SELECT body FROM home_items').fetchone()
        self.assertEqual(json.loads(item['body'])['contact_ready'], 0)

    def test_seller_edits_survive_draft_retry_and_contact_forgetting(self):
        from agents import revenue
        self.mission()
        self.drain()
        child = self.con.execute('SELECT * FROM revenue_children').fetchone()
        revenue.edit_email(self.con, child['id'], 'Aisha Tan', 0, 'My subject', 'My edited email')
        with self.assertRaises(ValueError):
            revenue.edit_email(self.con, child['id'], 'Aisha Tan', 0, 'Stale tab', 'Overwrite')
        revenue.retry(self.con, child['id'], 'Aisha Tan')
        self.drain()
        child = self.con.execute('SELECT * FROM revenue_children').fetchone()
        self.assertEqual(child['email_body'], 'My edited email')
        self.assertTrue(revenue.child_view(self.con, child['id'], 'Aisha Tan')['contact_missing'])

    def test_home_has_no_composer_and_autopilot_has_bulk_saved_progress(self):
        from agents import revenue
        self.mission()
        page = TestClient(web.app).get('/').text
        self.assertNotIn('class="home-voice"', page)
        self.assertNotIn('name="q" rows="1"', page)
        self.assertNotIn('revenue-mission"', page)
        self.assertIn('Agent set up',TestClient(web.app).get('/autopilot').text)
        self.assertIn('name="q"', TestClient(web.app).get('/ask').text)
        revenue.work_once(self.con,self.g)
        cid=self.con.execute('SELECT id FROM revenue_children').fetchone()[0]
        self.assertEqual(TestClient(web.app).get(f'/revenue/company/{cid}').status_code,200)


class KnowledgeTest(RevenueBase):
    def test_quotes_allow_pdf_line_wraps_but_reject_paraphrases(self):
        from agents import revenue
        def research(quote):
            return revenue.Research(why_now=[revenue.Claim(text='A sourced expansion.',source_id=1,quote=quote,kind='fact')],renewals=[],competitors=[],approach='Ask about the expansion.',unknowns=[])
        docs=[{'id':1,'text':'The company opened a new\n  distribution hub. Existing operations continue.'}]
        revenue.validate_research(research('opened a new distribution hub.'),docs)
        with self.assertRaises(revenue.CitationRejected):
            revenue.validate_research(research('launched a new distribution hub.'),docs)
        with self.assertRaises(revenue.CitationRejected):
            revenue.validate_research(research('opened a new Existing operations'),docs)
        with self.assertRaises(revenue.CitationRejected):
            revenue.validate_research(research(' \n\t '),docs)

    def test_citation_retry_receives_saved_validation_reason(self):
        from agents import revenue
        self.mission()
        with mock.patch('backend.knowledge.collect',self.fake_sources):
            for _ in range(3):revenue.work_once(self.con,self.g)
        job=revenue.claim(self.con)
        self.assertEqual(job['stage'],'research')
        error=revenue.CitationRejected('Source 1: quote must be a short continuous verbatim excerpt.')
        revenue.fail(self.con,job,error,revenue.now())
        next_at=datetime.fromisoformat(self.con.execute('SELECT next_at FROM revenue_children').fetchone()[0])
        retry=revenue.claim(self.con,next_at)
        contexts=[]
        def model(schema,system,user,**kw):
            contexts.append(json.loads(user))
            return self.fake_model(schema,system,user,**kw)
        with mock.patch.object(llm,'live',return_value=True),mock.patch.object(llm,'ask',model):
            revenue.perform(self.con,self.g,retry,next_at)
        self.assertEqual(contexts[0]['previous_validation_error'],str(error))

    def test_fetch_rejects_private_addresses_and_redirects(self):
        from backend import knowledge
        for u in ('http://127.0.0.1/x', 'http://169.254.169.254/', 'http://user:pw@example.com', 'file:///etc/passwd', 'https://example.com:444/'):
            with self.subTest(url=u), self.assertRaises(ValueError):
                knowledge.fetch(u)

    def test_new_entity_survives_absent_deployed_graph(self):
        from backend import graph
        self.mission()
        self.drain()
        self.g.execute('DELETE FROM entities WHERE uen=?', (SP,))
        self.g.commit()
        p = graph.profile(self.g, SP, app=self.con)
        self.assertEqual(p['e']['uen'], SP)
        self.assertEqual(TestClient(web.app).get('/company/' + SP).status_code, 200)

    def test_research_rejects_unsupported_citations(self):
        from agents import revenue
        bad = revenue.Research(why_now=[revenue.Claim(text='Ends next week', source_id=9, quote='not in source', kind='fact')],
                               renewals=[], competitors=[], approach='Ask', unknowns=[])
        with self.assertRaises(ValueError):
            revenue.validate_research(bad, [{'id':9,'text':'Registry information'}])


class ConsentTest(RevenueBase):
    def grant(self, child, person):
        from agents import revenue_contacts
        return revenue_contacts.grant(self.con,child['id'],'Aisha Tan',person['id'],(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())

    def test_global_people_budget_is_atomic(self):
        from concurrent.futures import ThreadPoolExecutor
        from backend import people
        def reserve(_):
            with db.connect() as c:
                return people.spend(c,1)
        with mock.patch.object(people,'CREDIT_CAP',1),ThreadPoolExecutor(2) as pool:
            result=list(pool.map(reserve,range(2)))
        self.assertEqual(sum(result),1)

    def test_model_zero_and_single_credit_budget(self):
        from concurrent.futures import ThreadPoolExecutor
        from agents import flows
        with mock.patch.object(llm,'live',return_value=True),mock.patch.object(llm,'ask',return_value='ok') as call:
            with mock.patch.object(flows,'CAP',0),self.assertRaises(llm.BudgetBlocked):
                llm.budgeted_ask(str,'s','u')
            call.assert_not_called()
            def ask(_):
                try:return llm.budgeted_ask(str,'s','u')
                except llm.BudgetBlocked:return 'blocked'
            with mock.patch.object(flows,'CAP',1),ThreadPoolExecutor(2) as pool:
                result=list(pool.map(ask,range(2)))
            self.assertEqual(result.count('ok'),1)
            self.assertEqual(call.call_count,1)

    def test_expired_and_zero_budget_grants_never_call_paid_adapter(self):
        from agents import revenue,revenue_contacts
        from backend import people
        child,p=self.prepare()
        gid=self.grant(child,p)
        with mock.patch.dict(os.environ,{'APOLLO_API_KEY':'test'}),mock.patch.object(people,'enrich',side_effect=AssertionError('No paid calls')):
            self.con.execute("UPDATE revenue_contact_grants SET expires='2000-01-01T00:00:00+00:00' WHERE id=?",(gid,))
            self.drain()
            self.assertEqual(self.count('revenue_reveal_attempts'),0)
            self.con.execute('UPDATE revenue_contact_grants SET expires=? WHERE id=?',((datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),gid))
            revenue.retry(self.con,child['id'],'Aisha Tan')
            with mock.patch.object(people,'CREDIT_CAP',0):
                self.drain()
            self.assertEqual(self.count('revenue_reveal_attempts'),0)
            self.assertEqual(self.con.execute('SELECT remaining FROM revenue_contact_grants').fetchone()[0],1)

    def test_ambiguous_paid_request_never_repeats_or_overwrites_edits(self):
        from agents import revenue
        from backend import people
        child,p=self.prepare()
        self.grant(child,p)
        revenue.edit_email(self.con,child['id'],'Aisha Tan',0,'My subject','My seller draft')
        with mock.patch.dict(os.environ,{'APOLLO_API_KEY':'test'}),mock.patch.object(people,'CREDIT_CAP',2),mock.patch.object(people,'enrich',side_effect=TimeoutError('uncertain')) as paid:
            self.drain()
            revenue.retry(self.con,child['id'],'Aisha Tan')
            self.drain()
        self.assertEqual(paid.call_count,1)
        self.assertEqual(self.con.execute('SELECT state FROM revenue_reveal_attempts').fetchone()[0],'uncertain')
        child=self.con.execute('SELECT * FROM revenue_children').fetchone()
        self.assertEqual(child['email_body'],'My seller draft')
        self.assertEqual(child['state'],'needs_approval')

    def test_wrong_owner_person_and_stop_block_reveal_and_results(self):
        from agents import revenue,revenue_contacts
        from backend import people
        child,p=self.prepare()
        expiry=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
        with self.assertRaises(ValueError):
            revenue_contacts.grant(self.con,child['id'],'Other Seller',p['id'],expiry)
        with self.assertRaises(ValueError):
            revenue_contacts.grant(self.con,child['id'],'Aisha Tan',1,expiry)
        self.grant(child,p)
        job=revenue.claim(self.con)
        reply=self.preview|{'name':'Ana Tan','email':'ana@ninjavan.co','email_status':'verified'}
        def stopped(_):
            revenue.stop(self.con,child['mission_id'],'Aisha Tan')
            return [reply]
        with mock.patch.dict(os.environ,{'APOLLO_API_KEY':'test'}),mock.patch.object(people,'CREDIT_CAP',1),mock.patch.object(people,'enrich',stopped):
            result=revenue_contacts.reveal(self.con,job,dict(self.con.execute('SELECT * FROM companies WHERE id=?',(child['company_id'],)).fetchone()),datetime.now(timezone.utc))
            with self.assertRaises(revenue.LostLease):
                revenue.complete(self.con,job,{'reveal':result})
        self.assertIsNone(self.con.execute('SELECT email FROM people WHERE id=?',(p['id'],)).fetchone()[0])

    def test_display_and_review_revalidate_deleted_or_stopped_contact(self):
        from agents import revenue
        from backend import people
        child,p=self.prepare()
        self.con.execute("UPDATE people SET email='ana@ninjavan.co',email_status='verified' WHERE id=?",(p['id'],))
        revenue.retry(self.con,child['id'],'Aisha Tan')
        self.drain()
        self.assertFalse(revenue.child_view(self.con,child['id'],'Aisha Tan')['contact_missing'])
        self.con.execute('DELETE FROM people WHERE id=?',(p['id'],))
        v=revenue.child_view(self.con,child['id'],'Aisha Tan')
        self.assertTrue(v['contact_missing'])
        self.assertEqual(v['artifact']['context']['people'],[])
        with self.assertRaises(ValueError):
            revenue.review_email(self.con,child['id'],'Aisha Tan')


class ConcurrencyTest(RevenueBase):
    def test_two_runs_reserve_distinct_companies_and_report_actual_counts(self):
        from agents import revenue
        from backend import graph
        from concurrent.futures import ThreadPoolExecutor
        mid=self.mission()
        due=self.con.execute('SELECT due_at FROM revenue_runs').fetchone()[0]
        with self.con.transaction():
            revenue.queue_run(self.con,mid,(datetime.fromisoformat(due)+timedelta(days=1)).isoformat(),datetime.now(timezone.utc))
        def run(_):
            with db.connect() as c,graph.readonly() as g:
                return revenue.work_once(c,g)
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(run,range(2)))
        children=self.con.execute('SELECT uen FROM revenue_children').fetchall()
        self.assertEqual({r[0] for r in children},{SP,NINJA})
        self.assertTrue(all(r[0] is None for r in self.con.execute('SELECT shortage FROM revenue_runs')))

    def test_parallel_late_completions_keep_both_home_cards(self):
        from agents import revenue
        from concurrent.futures import ThreadPoolExecutor
        self.mission(limit=2)
        self.drain(count=20)
        self.con.execute("UPDATE revenue_children SET state='queued',stage='draft',email_body=NULL")
        due=datetime.fromisoformat(self.con.execute('SELECT due_at FROM revenue_runs ORDER BY id LIMIT 1').fetchone()[0])
        revenue.maintain(self.con,due)
        jobs=[revenue.claim(self.con),revenue.claim(self.con)]
        self.assertTrue(all(jobs))
        def finish(job):
            with db.connect() as c:
                revenue.complete(c,job,{'email':{'subject':job['name'],'body':'Hi, a specific proposal.','source_ids':[]},'person_id':None})
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(finish,jobs))
        body=json.loads(self.con.execute('SELECT body FROM home_items ORDER BY id LIMIT 1').fetchone()[0])
        self.assertEqual(body['researched'],2)
        self.assertEqual(len(body['companies']),2)

    def test_stop_during_fetch_prevents_all_late_evidence(self):
        from agents import revenue
        mid=self.mission()
        revenue.work_once(self.con,self.g)
        def stopped(*args,**kw):
            revenue.stop(self.con,mid,'Aisha Tan')
            return self.fake_sources(*args,**kw)
        with mock.patch('backend.knowledge.collect',stopped):
            revenue.work_once(self.con,self.g)
        self.assertEqual(self.count('knowledge_documents'),0)
        self.assertEqual(self.con.execute('SELECT state FROM revenue_children').fetchone()[0],'cancelled')

    def test_known_company_reuses_monitor_without_repeating_history_fetch(self):
        from agents import revenue
        self.mission()
        self.drain()
        self.mission()
        with mock.patch('backend.knowledge.collect',side_effect=AssertionError('Fresh monitor should be reused')):
            self.drain()
        self.assertEqual(self.count('knowledge_monitors'),1)
        self.assertEqual(self.count('knowledge_subscriptions'),2)

    def test_transient_selection_failure_can_retry_and_owner_scope_is_kept(self):
        from agents import revenue
        mid=self.mission(uens=[NINJA])
        original=kg._record
        with mock.patch.object(kg,'_record',side_effect=OSError('transient graph read')):
            revenue.work_once(self.con,self.g)
        self.assertEqual(self.con.execute('SELECT state FROM revenue_runs').fetchone()[0],'queued')
        self.con.execute('UPDATE companies SET uen=? WHERE id=?',(NINJA,self.quiet))
        revenue.work_once(self.con,self.g,at=datetime.now(timezone.utc)+timedelta(seconds=31))
        self.assertEqual(self.count('revenue_children'),0)
        self.assertIn('Only 0',self.con.execute('SELECT shortage FROM revenue_runs').fetchone()[0])


class DeadlinePrivacyTest(RevenueBase):
    def test_delivery_clock_publishes_partial_while_collector_is_blocked(self):
        from agents import revenue
        import threading,time
        self.mission()
        due=datetime.now(timezone.utc)+timedelta(seconds=1)
        self.con.execute('UPDATE revenue_runs SET due_at=?',(due.isoformat(),))
        entered,release,stop=threading.Event(),threading.Event(),threading.Event()
        def collect(*args,**kw):
            entered.set()
            release.wait(10)
            return self.fake_sources(*args,**kw)
        with mock.patch('backend.knowledge.collect',collect):
            worker=threading.Thread(target=revenue.worker,args=(stop,),daemon=True)
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                end=time.monotonic()+8
                while time.monotonic()<end and not self.count('home_items'):
                    time.sleep(.05)
                self.assertEqual(self.count('home_items'),1)
                body=json.loads(self.con.execute('SELECT body FROM home_items').fetchone()[0])
                self.assertEqual((body['researched'],body['pending']),(0,1))
                self.assertFalse(release.is_set())
            finally:
                stop.set();release.set();worker.join(10)
                self.assertFalse(worker.is_alive())

    def test_forgetting_serializes_with_new_discovery_insert(self):
        from agents import revenue_contacts
        from backend import privacy
        from concurrent.futures import ThreadPoolExecutor,TimeoutError
        import threading
        co=dict(self.con.execute('SELECT * FROM companies WHERE id=?',(self.sp,)).fetchone())
        person={'id':'forget-during-import','first_name':'New','last_name_obfuscated':'P***','title':'CIO','organization_id':'org',
                'organization_name':co['name'],'domain':'singpost.com'}
        checked,release,forget_started=threading.Event(),threading.Event(),threading.Event()
        original=privacy.forgotten
        def inspect(c,ids):
            value=original(c,ids)
            if person['id'] in ids:
                checked.set();release.wait(5)
            return value
        def insert():
            with db.connect() as c,c.transaction():
                revenue_contacts.save_discovered(c,co,{'people':[person]},datetime.now(timezone.utc))
        def forget():
            forget_started.set()
            with db.connect() as c:
                return privacy.forget(c,person['id'])
        with mock.patch.object(privacy,'forgotten',inspect),ThreadPoolExecutor(2) as pool:
            first=pool.submit(insert)
            self.assertTrue(checked.wait(5))
            second=pool.submit(forget)
            self.assertTrue(forget_started.wait(5))
            try:
                with self.assertRaises(TimeoutError):second.result(timeout=.15)
            finally:
                release.set()
            first.result(timeout=5);second.result(timeout=5)
        self.assertEqual(self.con.execute('SELECT count(*) FROM people WHERE ext_id=?',(person['id'],)).fetchone()[0],0)
        self.assertIn(person['id'],privacy.forgotten(self.con,[person['id']]))

    def test_grant_expiry_normalizes_singapore_offset(self):
        from agents import revenue_contacts
        child,p=self.prepare()
        expires=(datetime.now(timezone.utc)+timedelta(hours=1)).astimezone(__import__('zoneinfo').ZoneInfo('Asia/Singapore'))
        gid=revenue_contacts.grant(self.con,child['id'],'Aisha Tan',p['id'],expires.isoformat())
        saved=self.con.execute('SELECT expires FROM revenue_contact_grants WHERE id=?',(gid,)).fetchone()[0]
        self.assertTrue(saved.endswith('+00:00'))
        self.assertEqual(datetime.fromisoformat(saved),expires)


class FinalAcceptanceTest(RevenueBase):
    def test_saved_progress_opens_after_twenty_newer_missions(self):
        first=self.mission()
        for _ in range(20):self.mission()
        response=TestClient(web.app).get('/revenue/'+str(first))
        self.assertEqual(response.status_code,200)

    def test_failed_monitor_does_not_satisfy_new_domain_primary_coverage(self):
        from agents import revenue
        child,p=self.prepare()
        self.con.execute("DELETE FROM knowledge_documents WHERE kind IN ('website','annual_report')")
        self.con.execute('UPDATE knowledge_monitors SET website_domain=NULL,refresh_at=?',(revenue.stamp(self.at-timedelta(seconds=1)),))
        self.con.execute("UPDATE revenue_children SET stage='collect',state='queued',domain_collected=NULL,next_at=?",(revenue.stamp(self.at),))
        def unavailable(*args,**kw):
            b=self.fake_sources(*args,**kw)
            b['documents']=[d for d in b['documents'] if d['kind']!='website']
            b['warnings']=['Primary website unavailable']
            return b
        with mock.patch('backend.knowledge.collect',unavailable):
            revenue.refresh_monitor(self.con,self.g,at=self.at)
        self.assertIsNone(self.con.execute('SELECT website_domain FROM knowledge_monitors').fetchone()[0])
        with mock.patch('backend.knowledge.collect',self.fake_sources) as collect:
            job=revenue.claim(self.con)
            result=revenue.perform(self.con,self.g,job,self.at)
            self.assertIn('bundle',result)
            revenue.complete(self.con,job,result)
        self.assertEqual(self.con.execute('SELECT website_domain FROM knowledge_monitors').fetchone()[0],'ninjavan.co')

    def test_grant_expiring_while_identity_lock_waits_never_dispatches(self):
        from agents import revenue,revenue_contacts
        from backend import people,privacy
        child,p=self.prepare()
        start=datetime.now(timezone.utc)
        end=start+timedelta(seconds=10)
        revenue_contacts.grant(self.con,child['id'],'Aisha Tan',p['id'],end.isoformat())
        job=revenue.claim(self.con)
        co=dict(self.con.execute('SELECT * FROM companies WHERE id=?',(child['company_id'],)).fetchone())
        clock=[start]
        original=privacy.identity_lock
        def delayed(c,extid):
            original(c,extid)
            clock[0]=end+timedelta(seconds=1)
        with mock.patch.dict(os.environ,{'APOLLO_API_KEY':'test'}),mock.patch.object(people,'CREDIT_CAP',1),mock.patch.object(people,'enrich') as paid,mock.patch.object(privacy,'identity_lock',delayed),mock.patch.object(revenue,'now',side_effect=lambda:clock[0]):
            result=revenue_contacts.reveal(self.con,job,co,start)
        paid.assert_not_called()
        self.assertEqual(result['status'],'needs_approval')
        self.assertEqual(self.con.execute('SELECT state FROM revenue_reveal_attempts').fetchone()[0],'cancelled')

    def test_grant_expiring_between_reservation_and_dispatch_never_calls_paid(self):
        from agents import revenue,revenue_contacts
        from backend import people
        child,p=self.prepare()
        start=datetime.now(timezone.utc)
        revenue_contacts.grant(self.con,child['id'],'Aisha Tan',p['id'],(start+timedelta(seconds=10)).isoformat())
        job=revenue.claim(self.con)
        co=dict(self.con.execute('SELECT * FROM companies WHERE id=?',(child['company_id'],)).fetchone())
        with mock.patch.dict(os.environ,{'APOLLO_API_KEY':'test'}),mock.patch.object(people,'CREDIT_CAP',1),mock.patch.object(people,'enrich',side_effect=AssertionError('Expired grant must not dispatch')) as paid, mock.patch.object(revenue,'now',side_effect=[start,start+timedelta(seconds=20),start+timedelta(seconds=20)]):
            result=revenue_contacts.reveal(self.con,job,co,start)
        paid.assert_not_called()
        self.assertEqual(result['status'],'needs_approval')
        self.assertEqual(self.con.execute('SELECT state FROM revenue_reveal_attempts').fetchone()[0],'cancelled')

    def test_new_domain_triggers_primary_collection_before_model_research(self):
        from agents import revenue
        from backend import people
        self.mission(uens=[NINJA])
        domains=[];contexts=[]
        def collect(g,uen,company,history=True):
            domains.append(company.get('domain'))
            b=self.fake_sources(g,uen,company,history)
            if company.get('domain'):
                b['documents'].append({'kind':'website','url':'https://ninjavan.co/about','title':'Official primary evidence','text':'NINJA LOGISTICS PTE. LTD. operates a new network hub.','published':None})
            return b
        def model(schema,system,user,**kw):
            contexts.append(json.loads(user))
            return self.fake_model(schema,system,user,**kw)
        result={'id':'new-domain-person','first_name':'Ana','last_name_obfuscated':'T***','title':'CTO','organization':{'name':'NINJA LOGISTICS PTE. LTD.','primary_domain':'ninjavan.co'}}
        with mock.patch.dict(os.environ,{'APOLLO_API_KEY':'test'}),mock.patch.object(people,'_post',return_value={'people':[result]}), mock.patch('backend.knowledge.collect',collect),mock.patch.object(llm,'live',return_value=True),mock.patch.object(llm,'ask',model):
            for _ in range(12):
                if revenue.work_once(self.con,self.g) is None:break
        self.assertEqual(domains,[None,'ninjavan.co'])
        self.assertTrue(contexts)
        self.assertTrue(any(d['kind']=='website' for d in contexts[0]['documents']))
        self.assertEqual(self.con.execute('SELECT stage FROM revenue_children').fetchone()[0],'done')

    def test_seller_delivery_and_grant_times_are_singapore_time(self):
        from agents import revenue
        self.mission()
        revenue.work_once(self.con,self.g)
        client=TestClient(web.app)
        self.assertTrue('SGT' in client.get('/autopilot').text,'Autopilot must show Singapore delivery time')
        self.assertTrue('SGT' in client.get('/revenue/1').text,'Progress must show Singapore delivery time')
        self.assertNotIn(' UTC</h2>',client.get('/revenue/1').text)


def sample_pdf(text):
    """One real PDF page, built without another runtime dependency."""
    content=('BT /F1 12 Tf 40 700 Td ('+text.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')+') Tj ET').encode()
    objs=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
          b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
          b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',b'<< /Length '+str(len(content)).encode()+b' >>\nstream\n'+content+b'\nendstream']
    data=b'%PDF-1.4\n';offsets=[]
    for i,obj in enumerate(objs,1):
        offsets.append(len(data));data+=str(i).encode()+b' 0 obj\n'+obj+b'\nendobj\n'
    start=len(data);data+=b'xref\n0 6\n0000000000 65535 f \n'
    data+=b''.join(f'{x:010} 00000 n \n'.encode() for x in offsets)
    return data+b'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n'+str(start).encode()+b'\n%%EOF\n'


class PdfCollectionTest(RevenueBase):
    def test_official_annual_pdf_is_extracted_and_keeps_source_and_page(self):
        from backend import knowledge
        pdf=sample_pdf('SINGAPORE POST LIMITED Annual Report 2026. Network modernisation supports our new distribution hub.')
        def fetch(url,**kw):
            if 'news.google.com' in url:
                return {'url':url,'type':'application/xml','text':'<rss/>'}
            if url.endswith('.pdf'):
                return {'url':url,'type':'application/pdf','text':'','content':pdf}
            return {'url':url,'type':'text/html','text':'<html><body>SINGAPORE POST LIMITED <a href="/annual-report-2026.pdf">Annual report 2026</a></body></html>'}
        with mock.patch.object(knowledge,'fetch',fetch):
            bundle=knowledge.collect(self.g,SP,{'name':'Singapore Post','domain':'singpost.com'})
        reports=[d for d in bundle['documents'] if d['kind']=='annual_report']
        self.assertTrue(reports)
        self.assertIn('Network modernisation',reports[0]['text'])
        self.assertIn('annual-report-2026.pdf#page=1',reports[0]['url'])
        self.assertIn('pages 1',reports[0]['title'])

    def test_failed_pdf_is_truthful_partial_coverage(self):
        from backend import knowledge
        def fetch(url,**kw):
            if 'news.google.com' in url:return {'url':url,'type':'text/xml','text':'<rss/>'}
            if url.endswith('.pdf'):return {'url':url,'type':'application/pdf','text':'','content':b'%PDF-broken'}
            return {'url':url,'type':'text/html','text':'Singapore Post <a href="/annual-report-2026.pdf">Annual report</a>'}
        with mock.patch.object(knowledge,'fetch',fetch):
            bundle=knowledge.collect(self.g,SP,{'name':'Singapore Post','domain':'singpost.com'})
        self.assertFalse(any(d['kind']=='annual_report' for d in bundle['documents']))
        self.assertTrue(any('annual report' in w.lower() for w in bundle['warnings']))
