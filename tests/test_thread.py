"""Ask Signal: the agent thread's tools, its tool-use loop, result cards, approvals and pages."""
import base64
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import anthropic
import httpx2

from agents import llm, thread
from backend import db, kg, people
from tests.helpers import Base, seed_graph
from tests.test_web import WebBase

SP = "199201623M"


class ToolBase(Base):
    def setUp(self):
        super().setUp()
        seed_graph(self.g)
        kg.build(self.g)
        kg.link(self.con, self.g)

    def call(self, name, args, tid=None):
        return thread.call(self.con, self.g, "Aisha Tan", tid, name, args)


class ToolTest(ToolBase):
    def test_find_company_carries_source(self):
        top = self.call("find_company", {"name": "sinpost"})["results"][0]
        self.assertEqual(top["uen"], SP)
        self.assertTrue(top["source"])

    def test_profile_and_people_never_carry_email(self):
        profile = self.call("company_profile", {"uen": SP})
        self.assertEqual(profile["results"][0]["account"]["owner"], "Aisha Tan")
        self.assertNotIn("@", json.dumps(profile))
        people = self.call("account_people", {"uen": SP})
        self.assertEqual({p["name"] for p in people["results"]}, {"Lee Ming", "Joel Chng"})
        self.assertNotIn("@", json.dumps(people))

    def test_unknown_uen_and_tool_say_not_found(self):
        self.assertEqual(set(self.call("company_profile", {"uen": "000000000X"})), {"error"})
        self.assertIn("error", self.call("no_such_tool", {}))
        self.assertIn("error", self.call("find_company", {}))

    def test_results_capped(self):
        self.g.executemany("INSERT INTO contracts VALUES(?,?,?,?,?,?,?,'Awarded')",
                           [(f"T-9{i:03}", f"Parcel delivery lot {i}", "Agency C", "2025-01-01", "SINGAPORE POST LIMITED",
                             db.key("SINGAPORE POST LIMITED"), 1000.0) for i in range(400)])
        self.g.commit()
        out = self.call("company_links", {"uen": SP})
        self.assertLessEqual(len(json.dumps(out)), thread.MAX_RESULT + 200)
        self.assertIs(out["truncated"], True)

    def test_lookalikes_same_industry_not_ours(self):
        for uen, name in (("201100008H", "SWIFT MAIL PTE. LTD."), ("201200009J", "PARCEL HUB PTE. LTD."),
                          ("201300010K", "PEER LOGISTICS PTE. LTD.")):
            self.g.execute(f"INSERT INTO entities VALUES({','.join('?' * 14)})",
                           (uen, name, db.key(name), "Local Company", None, "Live Company", "2011-01-01", "53100",
                            "POSTAL ACTIVITIES", None, None, None, None, None))
        self.g.execute("INSERT INTO contracts VALUES('T-200','Mail room','Agency B','2025-05-01',?,?,5000,'Awarded')",
                       ("PARCEL HUB PTE. LTD.", db.key("PARCEL HUB PTE. LTD.")))
        self.g.commit()
        kg.build(self.g)
        self.con.execute("UPDATE companies SET uen='201300010K' WHERE id=?", (self.ids["quiet"],))  # already in the app
        self.con.commit()
        rows = kg.lookalikes(self.g, self.con, [SP])
        self.assertEqual([r["uen"] for r in rows], ["201200009J", "201100008H"])
        self.assertEqual(rows[0]["awards"], 1)
        for r in rows:
            self.assertIn("SINGAPORE POST", r["why"].upper())
            self.assertIn("POSTAL", r["why"].upper())
            self.assertTrue(r["source"])


def use(name, args, i="t1"):
    return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": i, "name": name, "input": args}]}


def say(text):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


class Fake:
    """Stands in for llm.agent_turn: plays the script in order (the last entry repeats) and keeps what it was sent."""
    def __init__(self, *script):
        self.script, self.got, self.systems = list(script), [], []

    def __call__(self, system, messages, tools):
        self.got.append(json.loads(json.dumps(messages)))
        self.systems.append(system)
        r = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(r, Exception):
            raise r
        return r


class LoopBase(ToolBase):
    def setUp(self):
        super().setUp()
        env = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"})
        env.start()
        self.addCleanup(env.stop)

    def model(self, *script):
        fake = Fake(*script)
        patch = mock.patch.object(llm, "agent_turn", fake)
        patch.start()
        self.addCleanup(patch.stop)
        return fake

    def ask(self, text, tid=None):
        tid = tid or thread.new(self.con, "Aisha Tan", text)
        thread.run(tid, text)
        return tid

    def steps(self, tid):
        return self.con.execute("SELECT * FROM steps WHERE thread_id=? ORDER BY id", (tid,)).fetchall()


class LoopTest(LoopBase):
    def test_turn_logs_work_and_answer(self):
        self.model(use("find_company", {"name": "sinpost"}), say("Singapore Post Limited."))
        tid = self.ask("Who is sinpost?")
        steps = self.steps(tid)
        self.assertEqual([s["kind"] for s in steps], ["user", "work", "answer"])
        self.assertEqual(steps[1]["text"], 'Looked up "sinpost" in the registry')
        self.assertIsNotNone(steps[1]["ms"])
        self.assertEqual(steps[2]["text"], "Singapore Post Limited.")
        t = self.con.execute("SELECT * FROM threads WHERE id=?", (tid,)).fetchone()
        self.assertEqual(t["state"], "idle")
        self.assertIn('"tool_result"', t["messages"])

    def test_follow_up_sends_whole_thread(self):
        fake = self.model(say("One."), say("Two."))
        tid = self.ask("First question")
        self.ask("Second question", tid)
        self.assertEqual(fake.got[1][0], {"role": "user", "content": "First question"})
        self.assertEqual(fake.got[1][-1], {"role": "user", "content": "Second question"})

    def test_agent_reads_gtm_memory(self):
        from agents import gtm
        gtm.seed(self.con)
        gtm.save(self.con, "icp", "Logistics firms with 500+ staff renewing a GeBIZ contract.")
        fake = self.model(say("Done."))
        self.ask("Who should I call?")
        self.assertTrue(fake.systems[0].startswith(llm.AGENT_SYSTEM))
        self.assertIn("Logistics firms with 500+ staff", fake.systems[0])

    def test_round_cap(self):
        fake = self.model(use("find_company", {"name": "sinpost"}))
        tid = self.ask("Loop forever")
        self.assertEqual(len(fake.got), thread.MAX_ROUNDS)
        self.assertEqual(thread.MAX_ROUNDS, 8)
        last = self.steps(tid)[-1]
        self.assertEqual(last["kind"], "stopped")
        self.assertIn("8 steps", last["text"])

    def test_model_error_stops_cleanly(self):
        self.model(anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")))
        tid = self.ask("Anything")
        self.assertEqual(self.steps(tid)[-1]["kind"], "stopped")
        self.assertEqual(self.con.execute("SELECT state FROM threads WHERE id=?", (tid,)).fetchone()[0], "idle")

    def test_no_key_refuses(self):
        self.model(say("never"))
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            tid = thread.new(self.con, "Aisha Tan", "Hello")
            with self.assertRaisesRegex(RuntimeError, "no key"):
                thread.run(tid, "Hello")
        self.assertEqual(self.steps(tid), [])


class CardTest(ToolBase):
    def setUp(self):
        super().setUp()
        self.tid = thread.new(self.con, "Aisha Tan", "Help me with SingPost")
        self.found = []
        patch = mock.patch.object(people, "find", lambda con, cid, product, k=3: self.found.append((cid, k)) or 2)
        patch.start()
        self.addCleanup(patch.stop)

    def last(self):
        return self.con.execute("SELECT * FROM steps WHERE thread_id=? ORDER BY id DESC", (self.tid,)).fetchone()

    def body(self, sid):
        return json.loads(self.con.execute("SELECT body FROM steps WHERE id=?", (sid,)).fetchone()[0])

    def use(self, name, args):
        return self.call(name, args, self.tid)

    def test_accounts_card_drops_unknown_rows(self):
        out = self.use("show_accounts", {"title": "Postal peers", "accounts": [
            {"uen": SP, "why": "Customer", "source": "CRM"}, {"uen": "999999999Z", "why": "Made up", "source": "CRM"}]})
        self.assertEqual(out["shown"], 1)
        card = self.last()
        self.assertEqual(card["kind"], "card")
        items = self.body(card["id"])["items"]
        self.assertEqual([i["uen"] for i in items], [SP])
        self.assertEqual(items[0]["name"], "SINGAPORE POST LIMITED")  # the graph's name, not the model's

    def test_brief_drops_unsourced_points(self):
        out = self.use("show_brief", {"uen": SP, "title": "SingPost in a minute", "points": [
            {"text": "Won two GeBIZ contracts this year", "source": "GeBIZ"}, {"text": "Is about to buy SD-WAN"}]})
        self.assertEqual(out["shown"], 1)
        b = self.body(self.last()["id"])
        self.assertEqual(b["type"], "brief")
        self.assertEqual([p["text"] for p in b["points"]], ["Won two GeBIZ contracts this year"])

    def people_ask(self):
        self.con.execute("UPDATE companies SET domain='singpost.com' WHERE id=?", (self.sp,))
        self.con.commit()
        out = self.use("request_people_lookup", {"uen": SP, "max_people": 2})
        return out["waiting"]

    def test_people_lookup_waits_then_runs_once(self):
        sid = self.people_ask()
        self.assertEqual(self.last()["state"], "waiting")
        self.assertEqual(self.body(sid)["cost"], "Up to 2 Apollo credits")
        self.assertEqual(self.found, [])
        thread.decide(self.con, sid, "approve")
        self.assertEqual(self.found, [(self.sp, 2)])
        self.assertEqual(self.last()["state"], "approved")
        self.assertEqual(self.body(sid)["result"], "Found 2 contacts")
        with self.assertRaises(ValueError):
            thread.decide(self.con, sid, "approve")
        self.assertEqual(len(self.found), 1)

    def test_people_lookup_needs_a_website(self):
        out = self.use("request_people_lookup", {"uen": SP})
        self.assertIn("website", out["error"])
        self.assertEqual(self.last(), None)

    def test_skip_never_acts(self):
        sid = self.people_ask()
        thread.decide(self.con, sid, "skip")
        self.assertEqual(self.last()["state"], "skipped")
        self.assertEqual(self.found, [])
        with self.assertRaises(ValueError):
            thread.decide(self.con, sid, "approve")
        self.assertEqual(self.found, [])

    def test_queue_drafts_approval_creates_tasks(self):
        self.con.execute("UPDATE companies SET uen='201000002B' WHERE id=?", (self.quiet,))
        self.con.commit()
        drafts = [{"uen": SP, "subject": "Parcel lockers", "body": "Hi Lee, write to joel@singpost.com"},
                  {"uen": "201000002B", "subject": "Fleet links", "body": "Hi there"}]
        out = self.use("request_queue_drafts", {"drafts": drafts})
        before = self.count("tasks")
        self.assertEqual(self.body(out["waiting"])["gets"], "2 drafts in your inbox, one per account")
        thread.decide(self.con, out["waiting"], "approve")
        rows = self.con.execute("SELECT * FROM tasks WHERE origin='agent' ORDER BY id").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(self.count("tasks"), before + 2)
        self.assertEqual({r["company_id"] for r in rows}, {self.sp, self.quiet})
        for r in rows:
            self.assertEqual(r["owner"], "Aisha Tan")
            self.assertEqual(r["ref"], f"thread:{self.tid}:{out['waiting']}")
            b = json.loads(r["body"])
            self.assertTrue(b["subject"] and b["draft"])
            self.assertNotIn("@", r["body"])
        self.assertEqual(self.body(out["waiting"])["result"], "2 drafts queued")

    def test_one_draft_reads_singular(self):
        out = self.use("request_queue_drafts", {"drafts": [{"uen": SP, "subject": "Hi", "body": "Hello"}]})
        self.assertEqual(self.last()["text"], "Put 1 draft in your inbox?")
        self.assertEqual(self.body(out["waiting"])["gets"], "1 draft in your inbox, one per account")

    def test_queue_refuses_prospect(self):
        out = self.use("request_queue_drafts", {"drafts": [{"uen": "201000002B", "subject": "Hi", "body": "Hello"}]})
        self.assertIn("not one of your accounts yet", out["error"])
        self.assertEqual(self.last(), None)

    def test_save_segment_once(self):
        self.use("show_accounts", {"title": "Postal peers", "accounts": [{"uen": SP, "why": "Customer", "source": "CRM"}]})
        sid = self.last()["id"]
        first = thread.save_segment(self.con, sid, "Aisha Tan")
        self.assertEqual(thread.save_segment(self.con, sid, "Aisha Tan"), first)
        self.assertEqual(self.count("segments"), 1)
        seg = self.con.execute("SELECT * FROM segments").fetchone()
        self.assertEqual((seg["owner"], seg["name"]), ("Aisha Tan", "Postal peers"))
        self.assertEqual([i["uen"] for i in json.loads(seg["items"])], [SP])

    def test_queue_draft_card_once(self):
        self.use("show_draft", {"uen": SP, "to_role": "Group CIO", "subject": "Parcel lockers", "body": "Hi Lee"})
        sid = self.last()["id"]
        first = thread.queue_draft(self.con, sid, "Aisha Tan")
        self.assertEqual(thread.queue_draft(self.con, sid, "Aisha Tan"), first)
        task = self.row(first)
        self.assertEqual((task["origin"], task["ref"], task["company_id"]), ("agent", f"thread:{self.tid}:{sid}", self.sp))
        self.assertEqual(json.loads(task["body"])["draft"], "Hi Lee")


    def test_build_workflow_saves_a_draft(self):
        spec = {"name": "Welcome new CIOs", "trigger": {"type": "signal", "signal": "leadership_change"},
                "steps": [{"type": "draft", "about": "a welcome note"}, {"type": "queue", "title": "Welcome note"}]}
        r = thread.call(self.con, self.g, "Aisha Tan", self.tid, "build_workflow", {"spec": spec})
        w = self.con.execute("SELECT * FROM workflows WHERE owner='Aisha Tan'").fetchone()
        self.assertEqual(w["name"], "Welcome new CIOs")
        self.assertEqual((w["state"], w["live_version"]), ("draft", None))  # drafts never fire
        b = self.body(r["card"])
        self.assertEqual((b["type"], b["workflow_id"]), ("workflow", w["id"]))
        from fastapi.testclient import TestClient
        from frontend import web
        html = TestClient(web.app).get(f"/ask/{self.tid}").text
        self.assertIn(f'href="/workflows/{w["id"]}"', html)

    def test_build_workflow_bad_spec_is_an_error(self):
        r = thread.call(self.con, self.g, "Aisha Tan", self.tid, "build_workflow", {"spec": {"steps": [{"type": "nope"}]}})
        self.assertIn("error", r)

class AskTest(WebBase):
    """The pages, the way the browser drives them: POST queues the turn, POST /run runs it and streams the steps. The
    model is a script."""
    def setUp(self):
        super().setUp()
        seed_graph(self.g)
        kg.build(self.g)
        kg.link(self.con, self.g)
        p = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"})
        p.start()
        self.addCleanup(p.stop)
        self.found = []
        p = mock.patch.object(people, "find", lambda con, cid, product, k=3: self.found.append(cid) or 2)
        p.start()
        self.addCleanup(p.stop)

    def model(self, *script):
        fake = Fake(*script)
        p = mock.patch.object(llm, "agent_turn", fake)
        p.start()
        self.addCleanup(p.stop)
        return fake

    def ask(self, q="Who is sinpost?", **more):
        r = self.web.post("/ask", data={"q": q, "as": "Aisha Tan"} | more)
        self.assertEqual(r.status_code, 303)
        self.run_turn(r.headers["location"])
        return r.headers["location"]

    def run_turn(self, path):
        """What the page does next: POST /run, which runs the queued turn and streams its steps until it is idle."""
        return self.web.post(path.split("#")[0] + "/run").text

    def last(self, kind=None):
        return self.con.execute("SELECT * FROM steps WHERE kind=coalesce(?, kind) ORDER BY id DESC", (kind,)).fetchone()

    def test_ask_without_key_says_so(self):
        """No model key: the seller's message and one plain line, never a scripted answer."""
        fake = Fake(say("never"))
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": ""}), \
                mock.patch.object(llm, "agent_turn", fake):
            self.assertTrue(self.ask("Brief me on sinpost").startswith("/ask/1"))
            page = self.web.get("/ask/1").text
        self.assertEqual(fake.got, [])
        self.assertEqual([s["kind"] for s in self.con.execute("SELECT kind FROM steps ORDER BY id")], ["user", "stopped"])
        self.assertIn("Chat needs a model key", page)
        self.assertIn('action="/ask/1"', page)  # the follow-up box is always there

    def test_post_queues_and_the_stream_runs_it(self):
        """POST only queues (the message shows at once); /run claims the turn once and streams whole rows, in order, each
        chunk closed by the mark the page splits on. A second /run has nothing to run."""
        fake = self.model(say("Done here."))
        r = self.web.post("/ask", data={"q": "Who is sinpost?", "as": "Aisha Tan"})
        self.assertEqual(self.con.execute("SELECT state FROM threads").fetchone()[0], "queued")
        self.assertEqual((self.last()["kind"], self.last()["state"]), ("user", "pending"))
        self.assertEqual(fake.got, [])
        page = self.web.get(r.headers["location"]).text
        self.assertIn('class="steps live"', page)
        body = self.run_turn(r.headers["location"])
        self.assertTrue(body.endswith("<!--/steps-->"))
        self.assertLess(body.index("k-user"), body.index("k-answer"))
        self.assertIn("Done here.", body)
        self.assertEqual(self.con.execute("SELECT state FROM threads").fetchone()[0], "idle")
        self.assertEqual(len(fake.got), 1)
        self.assertEqual(self.web.post(f"/ask/1/run?after={self.last()['id']}").text, "")  # idle, nothing new
        self.assertEqual(len(fake.got), 1)

    def test_pick_goes_back_to_the_model(self):
        self.model(use("ask_seller", {"question": "Which product first?", "options": ["SD-WAN", "Security"]}),
                   say("Pick one."), say("Security it is."))
        self.ask("Meet logistics CIOs")
        sid = self.last("ask")["id"]
        r = self.web.post(f"/ask/step/{sid}/pick", data={"v": "Security"})
        self.assertEqual(r.status_code, 303)
        self.run_turn(r.headers["location"])
        self.assertEqual(self.web.post(f"/ask/step/{sid}/pick", data={"v": "Security"}).status_code, 303)  # once
        self.assertIn("Security it is.", self.web.get("/ask/1").text)
        self.assertEqual([s[0] for s in self.con.execute("SELECT text FROM steps WHERE kind='user' ORDER BY id")],
                         ["Meet logistics CIOs", "Security"])

    def test_tags_and_files_reach_the_model(self):
        """@ tags arrive as context lines the model reads, files as file parts; the page shows both as chips."""
        fake = self.model(say("Noted."))
        csv = "data:text/csv;base64," + base64.b64encode(b"company\nAcme Freight\n").decode()
        self.ask("Summarise this", ref=[f"company:{self.sp}", "company:99999"],
                 file=[json.dumps({"name": "leads.csv", "data": csv})])
        content = fake.got[0][-1]["content"]
        f = next(b for b in content if b["type"] == "file")
        self.assertEqual((f["name"], f["media_type"]), ("leads.csv", "text/csv"))
        self.assertEqual(llm._file(f), ("text", "File leads.csv:\ncompany\nAcme Freight\n"))
        words = next(b["text"] for b in content if b["type"] == "text")
        self.assertIn("Singapore Post", words)
        self.assertIn(SP, words)
        self.assertTrue(words.rstrip().endswith("Summarise this"))
        self.assertEqual(len(json.loads(self.last("user")["body"])["refs"]), 1)  # the missing company is dropped
        page = self.web.get("/ask/1").text
        self.assertIn("leads.csv", page)
        self.assertIn('class="chip-t"', page)

    def test_files_check_before_the_thread_exists(self):
        bad = json.dumps({"name": "x.exe", "data": "data:application/octet-stream;base64,AAAA"})
        self.assertEqual(self.web.post("/ask", data={"q": "Read this", "as": "Aisha Tan", "file": bad}).status_code, 415)
        big = json.dumps({"name": "a.csv", "data": "data:text/csv;base64," + base64.b64encode(b"x" * 3_100_000).decode()})
        self.assertEqual(self.web.post("/ask", data={"q": "Read this", "as": "Aisha Tan", "file": big}).status_code, 413)
        self.assertEqual(self.con.execute("SELECT count(*) FROM threads").fetchone()[0], 0)

    def test_gemini_reads_files_as_parts(self):
        pdf = Path(self.enterContext(tempfile.TemporaryDirectory())) / "a.pdf"
        pdf.write_bytes(b"%PDF-1.4 tiny")
        msgs = [{"role": "user", "content": [{"type": "file", "name": "a.pdf", "media_type": "application/pdf",
                                               "path": str(pdf), "size": 13}, {"type": "text", "text": "Read it"}]}]
        parts = llm._gemini_contents(msgs)[0]["parts"]
        self.assertEqual(parts[0]["inlineData"], {"mimeType": "application/pdf",
                                                  "data": base64.b64encode(b"%PDF-1.4 tiny").decode()})
        self.assertEqual(parts[-1], {"text": "Read it"})
        gone = [{"role": "user", "content": [msgs[0]["content"][0] | {"path": "/nope/a.pdf"}]}]
        self.assertIn("no longer available", llm._gemini_contents(gone)[0]["parts"][0]["text"])

    def test_mention_puts_crm_accounts_first(self):
        self.con.execute("UPDATE companies SET in_crm=1 WHERE id=?", (self.sp,))
        self.con.commit()
        rows = self.web.get("/ask/mention?q=s&as=Aisha Tan").json()
        self.assertLessEqual(len(rows), 8)
        crm = [r["sub"] == "Your account" for r in rows if r["kind"] == "company"]
        self.assertTrue(crm and crm[0])
        self.assertEqual(crm, sorted(crm, reverse=True))
        self.assertTrue(all(r["ref"].startswith(r["kind"] + ":") for r in rows))

    def test_no_model_name_on_any_page(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": ""}):
            self.ask("Book 5 meetings with high-intent accounts this quarter")
            for path in ("/ask/1", "/ask?as=Aisha Tan", "/"):
                self.assertNotIn("Claude", self.web.get(path).text, path)

    def test_composer_suggests_workable_goals(self):
        """One starter per kind of goal, three at most, each one the seller's own counts."""
        page = self.web.get("/ask?as=Aisha Tan").text
        self.assertIn('href="/memory"', page)
        kinds, shown = set(), []
        for goal in thread.suggestions(self.con, "Aisha Tan"):
            if goal["intent"] not in kinds:
                kinds.add(goal["intent"])
                shown.append(goal)
        for goal in shown[:3]:
            self.assertIn(goal["text"], page)
        self.assertEqual(page.count("data-start="), 3)

    def test_ask_runs_and_shows_steps(self):
        self.model(use("find_company", {"name": "sinpost"}), say("Singapore Post Limited."))
        self.assertTrue(self.ask().startswith("/ask/1"))
        page = self.web.get("/ask/1").text
        for text in ("Who is sinpost?", "in the registry", "Singapore Post Limited."):
            self.assertIn(text, page)
        self.assertNotIn('class="steps live"', page)  # finished: nothing streams
        self.assertIn("Who is sinpost?", self.web.get("/ask?as=Aisha Tan").text)  # listed under the threads

    def test_approval_buttons_and_double_post(self):
        self.con.execute("UPDATE companies SET domain='singpost.com' WHERE id=?", (self.sp,))
        self.con.commit()
        self.model(use("request_people_lookup", {"uen": SP}), say("Approve it and I will look."))
        self.ask("Who should I talk to at SingPost?")
        page = self.web.get("/ask/1").text
        for text in ("Waiting for you", "Up to 3 Apollo credits", "Approve", "Skip"):
            self.assertIn(text, page)
        sid = self.last("approval")["id"]
        self.assertIn(f'action="/ask/step/{sid}/approve"', page)
        self.assertEqual(self.found, [])  # nothing happens until the approval
        for _ in range(2):
            r = self.web.post(f"/ask/step/{sid}/approve")
            self.assertEqual(r.status_code, 303)
            self.assertTrue(r.headers["location"].startswith("/ask/1"))
        self.assertEqual(self.found, [self.sp])
        page = self.web.get("/ask/1").text
        self.assertIn("Approved", page)
        self.assertIn("Found 2 contacts", page)
        self.assertNotIn(f'action="/ask/step/{sid}/approve"', page)

    def test_message_is_escaped(self):
        self.model(say("<b>bold</b> answer"))
        self.ask("<script>x</script>")
        page = self.web.get("/ask/1").text
        self.assertIn("&lt;script&gt;x&lt;/script&gt;", page)
        self.assertNotIn("<script>x</script>", page)
        self.assertIn("&lt;b&gt;bold&lt;/b&gt;", page)

    def test_stale_running_thread_recovers(self):
        tid = thread.new(self.con, "Aisha Tan", "Hello")
        thread.step(self.con, tid, "user", "Hello")
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(timespec="seconds")
        self.con.execute("UPDATE steps SET at=?", (old,))
        self.con.execute("UPDATE threads SET state='running', updated=?", (old,))
        self.con.commit()
        page = self.web.get(f"/ask/{tid}").text
        self.assertIn("stopped before it finished", page)
        self.assertEqual(self.con.execute("SELECT state FROM threads").fetchone()[0], "idle")

    def test_fresh_running_thread_refuses_follow_up(self):
        tid = thread.new(self.con, "Aisha Tan", "Hello")
        thread.step(self.con, tid, "user", "Hello")
        self.con.execute("UPDATE threads SET state='running'")
        self.con.commit()
        page = self.web.get(f"/ask/{tid}").text
        self.assertIn('data-state="running"', page)
        self.assertIn('class="working" role="status"', page)
        self.assertEqual(self.web.post(f"/ask/{tid}", data={"q": "And?"}).status_code, 409)

    def test_list_card_links_sources_and_saves(self):
        self.model(use("show_accounts", {"title": "Postal peers", "accounts": [
            {"uen": SP, "why": "Two GeBIZ awards", "source": "https://www.gebiz.gov.sg/x"},
            {"uen": "201000002B", "why": "Same trade", "source": "ACRA"}]}), say("Here they are."))
        self.ask("Find postal companies")
        page = self.web.get("/ask/1").text
        self.assertIn('href="https://www.gebiz.gov.sg/x"', page)
        self.assertIn('href="/company/201000002B"', page)
        sid = self.last("card")["id"]
        self.assertEqual(self.web.post(f"/ask/step/{sid}/save").status_code, 303)
        home = self.web.get("/ask?as=Aisha Tan").text
        self.assertIn("Saved lists", home)
        self.assertIn("Postal peers", home)

    def test_draft_card_queues_to_inbox(self):
        self.model(use("show_draft", {"uen": SP, "to_role": "Group CIO", "subject": "Lockers", "body": "Hi Lee"}),
                   say("Drafted."))
        self.ask("Draft a note to SingPost")
        sid = self.last("card")["id"]
        self.assertIn(f'action="/ask/step/{sid}/queue"', self.web.get("/ask/1").text)
        self.web.post(f"/ask/step/{sid}/queue")
        self.assertEqual(self.con.execute("SELECT count(*) FROM tasks WHERE origin='agent'").fetchone()[0], 1)
        self.assertIn("In inbox", self.web.get("/ask/1").text)

    def test_demo_calls_the_model(self):
        fake = Fake(say("Done."))
        db.kv_set(self.con, "demo", True)
        self.con.commit()
        with mock.patch.object(llm, "agent_turn", fake):  # agents are real on the public copy too (user, 2026-11-08)
            self.ask("Brief me on sinpost")
        self.assertTrue(fake.got)
        self.assertNotIn("Nothing is sent.", self.web.get("/ask").text)  # no demo banner (user, 2026-11-08)


class ModelJourneyTest(LoopBase):
    """The model's own cards: ask_seller, set_goal, and a publish card for workflows. No plan card any more."""
    def test_question_tool(self):
        self.model(use("ask_seller", {"question": "Which product first?", "options": ["SD-WAN", "Security"]}, "c"),
                   say("Pick one."))
        tid = self.ask("Meet logistics CIOs")
        self.assertEqual([s["kind"] for s in self.steps(tid)], ["user", "ask", "answer"])
        ask = self.steps(tid)[1]
        self.assertEqual(ask["state"], "waiting")
        self.assertEqual(thread.pick(self.con, ask["id"], "Security"), {"label": "Security", "value": "Security"})

    def test_set_goal_and_publish_card(self):
        tid = thread.new(self.con, "Aisha Tan", "x")
        r = thread.call(self.con, self.g, "Aisha Tan", tid, "set_goal",
                        {"metric": "replies", "target": 5, "deadline": "2099-12-31"})
        self.assertEqual(self.con.execute("SELECT target FROM goals").fetchone()[0], 5)
        self.assertIn("card", r)
        spec = {"name": "Weekly", "trigger": {"type": "schedule", "every": "week", "day": 0, "hour": 8},
                "steps": [{"type": "graph", "what": "news"}]}
        r = thread.call(self.con, self.g, "Aisha Tan", tid, "build_workflow", {"spec": spec})
        ok = self.con.execute("SELECT * FROM steps WHERE id=?", (r["waiting"],)).fetchone()
        self.assertEqual((ok["kind"], ok["state"], json.loads(ok["body"])["action"]), ("approval", "waiting", "publish"))
        thread.decide(self.con, ok["id"], "approve")
        self.assertEqual(self.con.execute("SELECT state FROM workflows").fetchone()[0], "live")

    def test_prompt_answers_in_markdown_and_asks(self):
        for words in ("ask_seller", "markdown"):
            self.assertIn(words, llm.AGENT_SYSTEM)
        for words in ("show_plan", "update_plan", "Claude"):
            self.assertNotIn(words, llm.AGENT_SYSTEM)
        self.assertNotIn("show_plan", [t["name"] for t in thread.TOOLS])

    def test_daily_cap_asks_before_a_turn(self):
        from agents import flows
        flows.spend(self.con, flows.CAP)
        tid = thread.new(self.con, "Aisha Tan", "Hello")
        with mock.patch.object(db, "spawn") as t:
            thread.start(tid, "Hello")
        t.assert_not_called()
        ok = self.steps(tid)[-1]
        self.assertEqual((ok["kind"], ok["state"], json.loads(ok["body"])["action"]), ("approval", "waiting", "budget"))
        thread.decide(self.con, ok["id"], "approve")  # approved: the turn queues for the page's stream
        self.assertEqual(self.con.execute("SELECT state FROM threads WHERE id=?", (tid,)).fetchone()[0], "queued")
        self.assertEqual(thread.claim(self.con, tid), ("Hello", {}))
