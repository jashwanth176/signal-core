import json
import sqlite3
import unittest

from backend import db, graph, kg
from tests import helpers
from tests.helpers import Base, seed_graph


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.g = helpers.fresh()[1]
        seed_graph(self.g)

    def test_build_links_lei_former_alias_supplier(self):
        n = kg.build(self.g)
        self.assertEqual((n["lei"], n["former"], n["alias"], n["gebiz_supplier"]), (4, 2, 2, 3))
        ids = lambda s: {tuple(r) for r in self.g.execute("SELECT value, uen, conf FROM ids WHERE scheme=?", (s,))}
        self.assertIn(("singpost", "199201623M", 1.0), ids("alias"))
        self.assertIn(("ninja van", "201000002B", 1.0), ids("alias"))
        self.assertIn(("speedpost international", "201500003C", 1.0), ids("former"))
        self.assertEqual(ids("gebiz_supplier"), {("SINGAPORE POST LIMITED", "199201623M", 1.0),
                                                 ("NAMU HOLDINGS PTE. LTD.", "202100005E", 1.0),
                                                 ("NAMU HOLDINGS", "202100005E", 0.5)})

    def test_build_twice_same_rows(self):
        kg.build(self.g)
        first = self.g.execute("SELECT count(*) FROM ids").fetchone()[0]
        self.assertEqual(kg.build(self.g)["names"], 8)
        self.assertEqual(self.g.execute("SELECT count(*) FROM ids").fetchone()[0], first)

    def test_build_without_row_factory(self):
        raw = sqlite3.connect(graph.PATH)
        self.assertEqual(kg.build(raw)["lei"], 4)


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.g = helpers.fresh()[1]
        seed_graph(self.g)
        kg.build(self.g)

    def top(self, q, **kw):
        r = kg.resolve(self.g, q, **kw)
        return (r[0]["uen"], r[0]["via"]) if r else None

    def test_exact_name_prefers_live(self):
        r = kg.resolve(self.g, "Singapore Post")
        self.assertEqual([x["uen"] for x in r], ["199201623M", "200000001A"])
        self.assertEqual((r[0]["via"], r[0]["score"]), ("exact", 1.0))

    def test_uen_typed_lowercase(self):
        self.assertEqual(self.top("199201623m"), ("199201623M", "uen"))

    def test_full_legal_name_beats_live(self):
        self.assertEqual(self.top("NAMU HOLDINGS PTE. LIMITED")[0], "202000004D")
        self.assertEqual(self.top("Namu Holdings")[0], "202100005E")

    def test_alias_trading_name(self):
        self.assertEqual(self.top("Ninja Van"), ("201000002B", "alias"))

    def test_former_name(self):
        self.assertEqual(self.top("Speedpost International"), ("201500003C", "former"))
        r = kg.resolve(self.g, "Red Kite Trading")  # someone else holds that name today: they come first
        self.assertEqual([(x["uen"], x["via"]) for x in r], [("201900007G", "exact"), ("201800006F", "former")])

    def test_typed_words_beat_key_collisions(self):
        # db.key drops Holdings/Corporation, so these collide on one key; the words the user typed decide
        for uen, name, former in (("199706229Z", "PSA CORPORATION LIMITED", None),
                                  ("197200399R", "PSA INTERNATIONAL PTE LTD", None),
                                  ("199604562C", "UMS PTE. LTD.", None),
                                  ("200100340R", "UMS INTEGRATION LIMITED", "UMS HOLDINGS"),
                                  ("201409279C", "PIL HOLDINGS PTE. LTD.", None),
                                  ("196700080N", "PACIFIC INTERNATIONAL LINES (PRIVATE) LIMITED", None)):
            self.g.execute("INSERT INTO entities(uen, name, key, status, former) VALUES(?,?,?,'Live Company',?)",
                           (uen, name, db.key(name), former))
        kg.build(self.g)
        self.assertEqual(self.top("PSA International")[0], "197200399R")
        self.assertEqual(self.top("UMS Holdings"), ("200100340R", "former"))  # renamed UMS Integration in 2024
        self.assertEqual(self.top("Pacific International Lines"), ("196700080N", "alias"))

    def test_curated_alias_typed_before_expansion(self):
        # db.key turns 'SingPost' into 'singapore post'; a struck-off shell named SINGPOST must not win on its spelling
        self.g.execute("INSERT INTO entities(uen, name, key, status) VALUES('202438662N', 'SINGPOST HOLDINGS PTE. LTD.', "
                       "?, 'Struck Off')", (db.key("SINGPOST HOLDINGS PTE. LTD."),))
        self.g.execute("INSERT INTO entities(uen, name, key, status) VALUES('T09000001X', 'NINJA VAN', 'ninja van', "
                       "'Cancelled')")  # a business name spelled exactly as typed: the curated alias still wins
        kg.build(self.g)
        self.assertEqual(self.top("SingPost")[0], "199201623M")
        self.assertEqual(self.top("Ninja Van")[0], "201000002B")

    def test_partial_name_never_beats_exact(self):
        for uen, name in (("199400001A", "THE ASCOTT LIMITED"), ("200500002B", "ASCOTT SINGAPORE RAFFLES PLACE PTE. LTD.")):
            self.g.execute("INSERT INTO entities(uen, name, key, status) VALUES(?,?,?,'Live Company')", (uen, name, db.key(name)))
        kg.build(self.g)
        self.assertEqual(self.top("Ascott", prefix=True), ("199400001A", "exact"))

    def test_lei_only_fund(self):
        r = kg.resolve(self.g, "Alpha Growth Fund VCC")[0]
        self.assertEqual((r["uen"], r["via"], r["name"], r["status"]), ("T21VC0001A", "lei", "ALPHA GROWTH FUND VCC", "ACTIVE"))

    def test_typos(self):
        self.assertEqual(self.top("Singapore Psot"), ("199201623M", "typo"))
        self.assertEqual(self.top("sinpost"), ("199201623M", "typo"))
        self.assertEqual(kg.resolve(self.g, "Singapore Psot")[0]["score"], 0.8)

    def test_abstains_on_unknown(self):
        self.assertEqual(kg.resolve(self.g, "Toyota Motor"), [])

    def test_prefix_only_when_asked(self):
        self.assertEqual(kg.resolve(self.g, "quantium sol"), [])
        self.assertEqual(self.top("quantium sol", prefix=True), ("201500003C", "prefix"))

    def test_odd_input_never_raises(self):
        for q in ("", "  ", "a", "ab", '"', 'AT&T "x" (S)* OR - NEAR', "新加坡邮政"):
            self.assertIsInstance(kg.resolve(self.g, q), list, q)
            self.assertIsInstance(kg.resolve(self.g, q, prefix=True), list, q)

    def test_unbuilt_graph_still_resolves_exact(self):
        g = helpers.fresh()[1]
        seed_graph(g)  # no kg.build: ids and entity_fts empty
        self.assertEqual(kg.resolve(g, "Singapore Post")[0]["uen"], "199201623M")


class EntityTest(unittest.TestCase):
    def setUp(self):
        self.g = helpers.fresh()[1]
        seed_graph(self.g)
        kg.build(self.g)

    def test_entity_lists_ids_names_and_sources(self):
        e = kg.entity(self.g, "199201623M")
        self.assertEqual((e["name"], e["status"], e["ssic_desc"]), ("SINGAPORE POST LIMITED", "Live Company", "POSTAL ACTIVITIES"))
        self.assertIn({"scheme": "lei", "value": "LEI0SINGPOST000000001", "source": "gleif", "conf": 1.0}, e["ids"])
        self.assertIn({"name": "singpost", "via": "alias"}, e["names"])
        self.assertEqual(e["as_of"]["gebiz"][:10], "2026-11-07")
        self.assertEqual(e["as_of"]["gdelt"][:10], "2026-11-08")

    def test_entity_lei_only_and_unknown(self):
        self.assertEqual(kg.entity(self.g, "T21VC0001A")["name"], "ALPHA GROWTH FUND VCC")
        self.assertIsNone(kg.entity(self.g, "NOPE"))

    def test_mentions_give_reason(self):
        m = {x["id"]: x["reason"] for x in kg.mentions(self.g, "199201623M")}
        self.assertEqual(set(m), {"n1", "n2"})  # n3 is not about Singapore
        self.assertIn("GDELT tagged it 'singapore post'", m["n1"])
        self.assertEqual(m["n2"], "headline names 'singpost'")
        self.assertEqual([x["id"] for x in kg.mentions(self.g, "199201623M", since="2026-11-02")], ["n2"])

    def test_mentions_keep_wire_copies_once(self):
        self.g.execute("INSERT INTO news VALUES('n4','2026-11-02T01:00:00','https://x.example/d','yahoo.com','eng','SN','','','',0,"
                       "'SingPost first-half profit rises')")
        self.assertEqual(len(kg.mentions(self.g, "199201623M")), 2)


class EdgeTest(unittest.TestCase):
    def setUp(self):
        self.app, self.g = helpers.fresh()
        self.sp = helpers.seed(self.app)["singpost"]
        seed_graph(self.g)
        kg.build(self.g)
        self.app.execute("UPDATE companies SET uen='199201623M' WHERE id=?", (self.sp,))

    def test_edges_ownership(self):
        self.assertEqual([(e["dst"], e["name"], e["source"]) for e in kg.edges(self.g, self.app, "201500003C", ["parent"])],
                         [("199201623M", "SINGAPORE POST LIMITED", "gleif")])
        self.assertEqual([e["dst"] for e in kg.edges(self.g, self.app, "199201623M", ["subsidiary"])], ["201500003C"])
        self.assertEqual(kg.edges(self.g, self.app, "201000002B", ["parent"])[0]["dst"], "lei:LEI0NINJAPARENT000004")

    def test_edges_contracts_carry_attribution(self):
        sp = kg.edges(self.g, self.app, "199201623M", ["awarded_by"])
        self.assertEqual([(e["dst"], e["at"], e["conf"]) for e in sp],
                         [("agency:Ministry of Testing", "2025-06-01", 1.0), ("agency:Ministry of Testing", "2025-03-01", 1.0)])
        self.assertEqual(sorted(e["conf"] for e in kg.edges(self.g, self.app, "202100005E", ["awarded_by"])), [0.5, 1.0])
        self.assertEqual(kg.edges(self.g, self.app, "202000004D", ["awarded_by"]), [])

    def test_edges_app_relations_and_alumni(self):
        peer = self.app.execute("SELECT id FROM companies WHERE key='peer logistics'").fetchone()[0]
        self.app.execute("INSERT INTO relations VALUES(?,?,?,?)", (self.sp, peer, "partner", "news"))
        self.app.execute("UPDATE people SET history=? WHERE ext_id='p2'", (json.dumps([
            {"title": "AVP", "organization_name": "Singapore Post", "end_date": None, "current": True},
            {"title": "CISO", "organization_name": "Ninja Logistics", "end_date": "2025-09-01", "current": False}]),))
        e = {x["kind"]: x for x in kg.edges(self.g, self.app, "199201623M")}
        self.assertEqual(e["partner"]["dst"], f"company:{peer}")
        self.assertEqual(e["alumni"]["dst"], "201000002B")
        self.assertIn("Joel Chng", e["alumni"]["note"])

    def test_edges_relation_reads_right_from_either_side(self):
        # stored as (src, dst, kind) = "dst is src's kind": Singapore Post --subsidiary--> Peer
        peer = self.app.execute("SELECT id FROM companies WHERE key='peer logistics'").fetchone()[0]
        self.app.execute("INSERT INTO relations VALUES(?,?,?,?)", (self.sp, peer, "subsidiary", "public"))
        self.app.execute("INSERT INTO relations VALUES(?,?,?,?)", (peer, self.sp, "supplier", "mock_crm"))
        mine = {x["kind"]: x["dst"] for x in kg.edges(self.g, self.app, "199201623M")}
        self.assertEqual(mine["subsidiary"], f"company:{peer}")
        self.assertEqual(mine["customer"], f"company:{peer}")  # SingPost supplies Peer, so Peer is its customer
        theirs = {x["kind"]: x["dst"] for x in kg.edges(self.g, self.app, f"company:{peer}")}
        self.assertEqual(theirs, {"parent": "199201623M", "supplier": "199201623M"})

    def test_edges_in_news_with_needs_two_articles(self):
        self.assertEqual(kg.edges(self.g, self.app, "199201623M", ["in_news_with"]), [])
        self.g.execute("INSERT INTO news VALUES('n5','2026-11-03T00:00:00','https://news.example/e','cna.com','eng','SN',"
                       "'singapore post;quantium solutions international','','',0,'Parcel firms merge')")
        self.g.execute("INSERT INTO mentions VALUES('singapore post','n5')")
        self.assertEqual([e["dst"] for e in kg.edges(self.g, self.app, "199201623M", ["in_news_with"])], ["201500003C"])

    def test_path(self):
        self.assertEqual([e["kind"] for e in kg.path(self.g, self.app, "201500003C", "199201623M")], ["parent"])
        self.assertEqual(kg.path(self.g, self.app, "199201623M", "199201623M"), [])
        self.assertIsNone(kg.path(self.g, self.app, "201500003C", "201000002B"))

    def test_path_survives_ownership_cycle(self):
        self.g.execute("INSERT INTO owns VALUES('LEI0SINGPOST000000001','LEI0QUANTIUM000000002','IS_DIRECTLY_CONSOLIDATED_BY','PUBLISHED')")
        self.assertIsNone(kg.path(self.g, self.app, "201500003C", "T21VC0001A"))


class LinkTest(Base):
    def setUp(self):
        super().setUp()
        seed_graph(self.g)
        kg.build(self.g)

    def test_link_stores_uen(self):
        self.assertGreaterEqual(kg.link(self.con, self.g), 1)
        uen = dict(self.con.execute("SELECT key, uen FROM companies").fetchall())
        self.assertEqual((uen["singapore post"], uen["quiet"]), ("199201623M", None))

    def test_graph_command_writes_stats(self):
        from backend.__main__ import main
        main("graph", "stats")
        self.assertEqual(db.kv_get(db.connect(), "graph_stats")["total"], graph.stats(self.g)["total"])



class TimingTest(unittest.TestCase):
    def test_only_future_contract_ends_set_the_timing(self):
        from datetime import date
        from backend import engine
        t = date(2026, 11, 9)
        news = {"type": "contract_ending", "date": "2026-10-01"}  # an article's date, not an end date
        crm = {"type": "contract_ending", "date": "2027-08-01"}
        self.assertEqual(engine.timing([(news, "contract_ending", 1)], t), (0.6, "No contract date known"))
        self.assertEqual(engine.timing([(news, "contract_ending", 1), (crm, "contract_ending", 1)], t)[1],
                         "Contract ends in ~9 months")


if __name__ == "__main__":
    unittest.main()
