"""GTM memory: shared sections with history, documents, suggestion cards with evidence, and the text Claude reads."""
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from agents import accounts, gtm, llm
from backend import catalog, db, engine
from frontend import web
from tests.helpers import Base, _card, ago

RESEARCH = " ".join((Path(__file__).resolve().parent.parent / "backend" / "seed" / "singtel-2026-11.md").read_text().split())


def decided(con, origin, cid, states, start=0):
    for i, st in enumerate(states, start):
        con.execute("INSERT INTO tasks(company_id, owner, origin, kind, ref, title, state, created, decided) "
                    "VALUES(?,?,?,?,?,?,?,?,?)", (cid, "Aisha Tan", origin, "x", f"x:{i}", "t", st, ago(20), ago(i)))
    con.commit()


class SectionTest(Base):
    def test_seed_once_in_order(self):
        s = gtm.sections(self.con)
        self.assertEqual([x["key"] for x in s], ["offer", "icp", "roles", "signals", "market", "products"])
        self.assertEqual(s[0]["title"], "Singtel Enterprise offer")
        for x in s:
            self.assertTrue(x["body"] and x["source"] and x["at"], x["key"])
        gtm.sections(self.con)
        self.assertEqual(self.count("gtm_versions"), 6)  # seeded once

    def test_seed_text_comes_from_sources(self):
        body = {x["key"]: x["body"] for x in gtm.sections(self.con)}
        for key in ("offer", "market"):
            for line in body[key].splitlines():
                self.assertIn(line.removeprefix("- "), RESEARCH, key)
        for label, about in catalog.SIGNALS.items():
            self.assertIn(about, body["signals"])
        for name in catalog.PRODUCTS.values():
            self.assertIn(name, body["products"])
        for _, label, _ in catalog.COMMITTEE:
            self.assertIn(label, body["roles"])
        self.assertIn("logistics 1.0", body["icp"])

    def test_edit_keeps_history(self):
        gtm.sections(self.con)
        gtm.save(self.con, "icp", "- Logistics groups with regional hubs")
        icp = next(x for x in gtm.sections(self.con) if x["key"] == "icp")
        self.assertEqual(icp["body"], "- Logistics groups with regional hubs")
        self.assertEqual(icp["source"], "Edited on this page")
        self.assertEqual(len(icp["history"]), 1)
        self.assertIn("logistics 1.0", icp["history"][0]["body"])

    def test_custom_sections_max_three(self):
        keys = [gtm.add_section(self.con, f"Notes {i}", "- a line") for i in range(3)]
        self.assertEqual(keys, ["custom-1", "custom-2", "custom-3"])
        with self.assertRaises(ValueError):
            gtm.add_section(self.con, "Fourth", "- one too many")
        gtm.save(self.con, "custom-2", "- renamed", title="Partner notes")
        s = gtm.sections(self.con)
        self.assertEqual([x["key"] for x in s][-3:], keys)
        self.assertEqual(s[-2]["title"], "Partner notes")
        self.assertTrue(s[-1]["custom"])

    def test_save_rejects_bad_input(self):
        gtm.sections(self.con)
        for key, body in (("nope", "- text"), ("icp", "   "), ("icp", "x" * 20001)):
            with self.assertRaises(ValueError):
                gtm.save(self.con, key, body)
        with self.assertRaises(ValueError):
            gtm.add_section(self.con, " ", "- text")

    def test_emails_are_stripped(self):
        gtm.sections(self.con)
        gtm.save(self.con, "market", "- Ask jane.doe@example.com.sg first")
        body = next(x for x in gtm.sections(self.con) if x["key"] == "market")["body"]
        self.assertNotRegex(body, r"\S+@\S+")
        self.assertIn("Ask", body)


class DocTest(Base):
    def test_docs_add_list_remove(self):
        a = gtm.add_doc(self.con, "win-themes.md", "# Themes\r\n- Renewals at agencies, ask ops@agency.gov.sg")
        b = gtm.add_doc(self.con, "targets.csv", "name,industry\nSATS,logistics")
        rows = gtm.docs(self.con)
        self.assertEqual([r["id"] for r in rows], [b, a])  # newest first
        self.assertEqual(rows[1]["name"], "win-themes.md")
        self.assertEqual(rows[1]["size"], len(rows[1]["body"].encode()))
        self.assertNotIn("@", rows[1]["body"])
        self.assertTrue(rows[1]["at"])
        gtm.remove_doc(self.con, a)
        self.assertEqual([r["id"] for r in gtm.docs(self.con)], [b])

    def test_doc_rejects_bad_type_and_size(self):
        for name, body in (("deck.pdf", "x"), ("notes.txt", " "), ("big.txt", "x" * 200001), ("", "x")):
            with self.assertRaises(ValueError, msg=name):
                gtm.add_doc(self.con, name, body)
        self.assertEqual(self.count("gtm_docs"), 0)


class SuggestTest(Base):
    BACKTEST = {"since": "2023-01-01", "data_end": "2026-03-28", "singtel_wins": 62, "renewal_wins": 19, "flagged": 14,
                "recall_on_renewals": 0.737, "median_lead_months": 6.55, "defend": 6, "attack": 8}

    def decided(self, origin, cid, states, start=0):
        decided(self.con, origin, cid, states, start)

    def card(self):
        self.decided("peer_breach", self.sp, ["dismissed"] * 6 + ["done"] * 2)
        gtm.suggest(self.con)
        return gtm.suggestions(self.con)[0]

    def body(self, key):
        return next(x for x in gtm.sections(self.con) if x["key"] == key)

    def test_dismissals_become_a_card(self):
        gtm.sections(self.con)
        self.decided("peer_breach", self.sp, ["dismissed"] * 6 + ["done"] * 2)
        self.assertEqual(gtm.suggest(self.con), 1)
        [s] = gtm.suggestions(self.con)
        self.assertEqual(s["headline"], "Briefing offers were dismissed 6 of 8 times at logistics accounts")
        self.assertEqual(s["section"], "signals")
        self.assertIn("6 dismissed", s["evidence"])
        self.assertIn("2 approved", s["evidence"])
        self.assertEqual(self.count("gtm_versions"), 6)  # nothing changes until approved
        self.assertEqual(gtm.suggest(self.con), 0)  # same evidence, no second card

    def test_approvals_become_a_card(self):
        self.decided("renewal_due", self.ag, ["done"] * 5)
        gtm.suggest(self.con)
        [s] = gtm.suggestions(self.con)
        self.assertEqual(s["headline"], "Renewal packs were approved 5 of 5 times at government accounts")

    def test_too_few_decisions_no_card(self):
        self.decided("peer_breach", self.sp, ["dismissed"] * 3)
        self.decided("card", self.quiet, ["dismissed", "done", "dismissed", "done"], start=10)  # mixed: no pattern
        self.assertEqual(gtm.suggest(self.con), 0)

    def test_approve_appends_once(self):
        s = self.card()
        gtm.decide(self.con, s["id"], "approve")
        sig = self.body("signals")
        self.assertTrue(sig["body"].endswith(s["text"]))
        self.assertEqual(sig["source"], "Suggestion approved: " + s["headline"])
        with self.assertRaises(ValueError):
            gtm.decide(self.con, s["id"], "approve")  # a double click applies it once
        self.assertEqual(len(self.body("signals")["history"]), 1)
        self.assertEqual(gtm.suggestions(self.con), [])
        self.assertTrue(gtm.suggestions(self.con, "approved")[0]["decided"])

    def test_edit_then_approve_uses_edited_text(self):
        s = self.card()
        gtm.decide(self.con, s["id"], "edit", text="- Logistics: lead with the renewal, not the breach")
        self.decided("peer_breach", self.sp, ["dismissed"], start=8)  # newer evidence keeps the seller's words
        self.assertEqual(gtm.suggest(self.con), 1)
        s = gtm.suggestions(self.con)[0]
        self.assertIn("7 of 9", s["headline"])
        self.assertEqual(s["text"], "- Logistics: lead with the renewal, not the breach")
        gtm.decide(self.con, s["id"], "approve", text="- Logistics: open with the renewal")
        self.assertTrue(self.body("signals")["body"].endswith("- Logistics: open with the renewal"))

    def test_dismissed_topic_never_returns(self):
        s = self.card()
        gtm.decide(self.con, s["id"], "dismiss")
        self.decided("peer_breach", self.sp, ["dismissed"] * 3, start=8)
        self.assertEqual(gtm.suggest(self.con), 0)
        self.assertEqual(gtm.suggestions(self.con), [])
        self.assertEqual(self.count("gtm_versions"), 0)  # a dismissal writes nothing to memory

    def test_backtest_card(self):
        db.kv_set(self.con, "backtest_gebiz", self.BACKTEST)
        self.assertEqual(gtm.suggest(self.con), 1)
        [s] = gtm.suggestions(self.con)
        self.assertEqual(s["headline"], "The renewal radar flagged 14 of 19 Singtel renewal wins on GeBIZ before the award")
        self.assertEqual(s["section"], "signals")
        for part in ("median 6.55 months", "6 we already held, 8 taken from another supplier", "2023-01-01 to 2026-03-28"):
            self.assertIn(part, s["evidence"] + s["text"])
        self.assertEqual(gtm.suggest(self.con), 0)
        db.kv_set(self.con, "backtest_gebiz", self.BACKTEST | {"data_end": "2026-09-30"})
        self.assertEqual(gtm.suggest(self.con), 1)  # a new run is new evidence

    def test_won_deals_card(self):
        peer = self.con.execute("SELECT id FROM companies WHERE key='peer logistics'").fetchone()[0]
        _card(self.con, peer, "Aisha Tan", 0)
        for cid in (self.sp, peer, peer):
            oid = self.con.execute("SELECT id FROM opps WHERE company_id=?", (cid,)).fetchone()[0]
            self.con.execute("INSERT INTO feedback(opp_id, kind) VALUES(?, 'won')", (oid,))
        self.assertEqual(gtm.suggest(self.con), 1)
        [s] = gtm.suggestions(self.con)
        self.assertEqual(s["headline"], "2 deals marked won at logistics accounts")
        self.assertEqual(s["section"], "icp")
        self.assertIn("Managed Security & SOC", s["evidence"])
        self.assertIn("Singapore Post", s["evidence"])


class ReaderTest(Base):
    def test_context_has_sections_and_docs_capped(self):
        gtm.add_doc(self.con, "themes.md", "- Lead with renewals at agencies")
        gtm.add_doc(self.con, "big.csv", "a,b\n" + "x,y\n" * 20000)
        text = gtm.context(self.con)
        self.assertLessEqual(len(text), 4000)
        for part in ("Singtel Enterprise offer", "Buyer roles", "Tech leader", "Lead with renewals at agencies", "big.csv"):
            self.assertIn(part, text)
        self.assertLessEqual(len(gtm.context(self.con, limit=500)), 500)

    def test_card_writer_reads_memory(self):
        gtm.sections(self.con)
        gtm.save(self.con, "icp", "- Ports and airports first")
        card = llm.Card(headline="h", why_now=[], customer_goal="g", pitch="p", contact_role="r", opener="o",
                        email_subject="s", email_body="b", risk="r")
        with mock.patch.object(llm, "write_card", return_value=card) as m:
            engine.cards(self.con)
        self.assertIn("Ports and airports first", m.call_args.args[0]["gtm_memory"])

    def test_draft_writer_reads_memory(self):
        gtm.sections(self.con)
        gtm.save(self.con, "market", "- Agencies buy through NCS panels")
        accounts.run(self.con, self.g)
        d = llm.Draft(to_role="Group CIO", subject="s", body="b", ids=[])
        with mock.patch.object(llm, "write_draft", return_value=d) as m:
            accounts.draft_pending(self.con)
        self.assertIn("Agencies buy through NCS panels", m.call_args.args[0]["gtm_memory"])


class PageTest(Base):
    def setUp(self):
        super().setUp()
        self.web = TestClient(web.app, follow_redirects=False)

    def test_page_shows_sections_with_source_and_date(self):
        r = self.web.get("/memory")
        self.assertEqual(r.status_code, 200)
        for part in ("Singtel Enterprise offer", "Ideal customer profile", "Buyer roles", "Buying signals",
                     "Singapore market", "Products", "From seed/singtel-2026-11.md", "From catalog.py SIGNALS",
                     "Changed " + web.T.env.filters["fmtdate"](gtm.now())):
            self.assertIn(part, r.text)

    def test_edit_route_saves(self):
        self.web.get("/memory")
        r = self.web.post("/memory/section/icp", data={"body": "- Ports first"})
        self.assertEqual((r.status_code, r.headers["location"]), (303, "/memory#icp"))
        page = self.web.get("/memory").text
        self.assertIn("- Ports first", page)
        self.assertIn("1 earlier version", page)
        r = self.web.post("/memory/section", data={"title": "Partner notes", "body": "- Bridge Alliance"})
        self.assertEqual(r.headers["location"], "/memory#custom-1")
        self.assertIn("Partner notes", self.web.get("/memory").text)
        self.assertEqual(self.web.post("/memory/section/nope", data={"body": "x"}).status_code, 400)
        self.assertEqual(self.web.post("/memory/section/icp", data={"body": ""}).status_code, 400)

    def test_doc_routes(self):
        r = self.web.post("/memory/docs", data={"name": "renewal-notes.md", "body": "- Lead with renewals"})
        self.assertEqual((r.status_code, r.headers["location"]), (303, "/memory#docs"))
        page = self.web.get("/memory").text
        self.assertIn("renewal-notes.md", page)
        self.assertIn("20 Bytes", page)  # jinja filesizeformat
        did = gtm.docs(self.con)[0]["id"]
        self.assertEqual(self.web.post(f"/memory/docs/{did}/remove").status_code, 303)
        self.assertNotIn("renewal-notes.md", self.web.get("/memory").text)
        self.assertEqual(self.web.post("/memory/docs", data={"name": "deck.pdf", "body": "x"}).status_code, 400)

    def test_suggestion_route(self):
        decided(self.con, "peer_breach", self.sp, ["dismissed"] * 6 + ["done"] * 2)
        page = self.web.get("/memory").text
        self.assertIn("Briefing offers were dismissed 6 of 8 times at logistics accounts</strong>: add a note under "
                      "buying signals?", page)
        self.assertIn("6 dismissed, 2 approved", page)
        sid = gtm.suggestions(self.con)[0]["id"]
        r = self.web.post(f"/memory/suggestion/{sid}", data={"action": "approve", "text": "- Logistics: lead with renewals"})
        self.assertEqual((r.status_code, r.headers["location"]), (303, "/memory#signals"))
        page = self.web.get("/memory").text
        self.assertIn("- Logistics: lead with renewals", page)
        self.assertNotIn("add a note under", page)
        self.assertEqual(self.web.post(f"/memory/suggestion/{sid}", data={"action": "approve"}).status_code, 400)

    def test_html_renders_as_text(self):
        self.web.get("/memory")
        self.web.post("/memory/section/icp", data={"body": "<script>alert(1)</script>"})
        self.web.post("/memory/docs", data={"name": "<b>x</b>.md", "body": "<img src=x onerror=alert(1)>"})
        page = self.web.get("/memory").text
        self.assertIn("&lt;script&gt;alert(1)", page)
        self.assertNotIn("<script>alert(1)", page)
        self.assertNotIn("<img src=x", page)
        self.assertNotIn("<b>x</b>", page)

    def test_demo_copy_is_editable(self):
        decided(self.con, "peer_breach", self.sp, ["dismissed"] * 6 + ["done"] * 2)
        db.kv_set(self.con, "demo", {"entities": 1})
        self.con.commit()
        r = self.web.get("/memory")
        self.assertEqual(r.status_code, 200)
        self.assertIn('action="/memory/section/', r.text)
        self.assertGreater(self.count("gtm_suggestions"), 0)  # suggestions are worked out on the demo too
        self.assertEqual(self.web.post("/memory/section/icp", data={"body": "- Ports first"}).status_code, 303)
        self.assertIn("Ports first", self.web.get("/memory").text)

if __name__ == "__main__":
    unittest.main()
