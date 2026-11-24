"""Throwaway databases for the tests: a fresh app DB (a Postgres schema) and graph DB per test, seeded with a few
real-shaped rows."""
import itertools
import json
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import psycopg

from backend import db, graph

BASE = os.environ["TEST_DATABASE_URL"]  # checked in tests/__init__.py; never db.URL, the app database
_n = itertools.count()


def _sweep():
    """Drop the schemas earlier test runs left behind (named t<pid>_<n>; a run still going keeps its own), and refuse a
    test database that would ignore the per-test schema."""
    with psycopg.connect(BASE, autocommit=True) as c:
        for (s,) in c.execute(r"SELECT nspname FROM pg_namespace WHERE nspname ~ '^t[0-9]+_[0-9]+$'").fetchall():
            try:
                os.kill(int(s[1:].split("_")[0]), 0)
            except ProcessLookupError:
                c.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
    with psycopg.connect(_url("t_probe"), autocommit=True) as c:  # a pooler drops it, and the tests would hit public
        if c.execute("SHOW search_path").fetchone()[0] != "t_probe":
            raise SystemExit("TEST_DATABASE_URL must be a direct connection: it ignores the per-test search_path")


def _url(schema):
    return f"{BASE}{'&' if '?' in BASE else '?'}options=-csearch_path%3D{schema}"


_sweep()


def ago(days):
    return (date.today() - timedelta(days=days)).isoformat()


def ahead(days):
    return (date.today() + timedelta(days=days)).isoformat()


def fresh():
    schema = f"t{os.getpid()}_{next(_n)}"
    with psycopg.connect(BASE, autocommit=True) as c:
        c.execute(f"CREATE SCHEMA {schema}")
    db.URL = _url(schema)
    db.DATA = Path(tempfile.mkdtemp())
    graph.PATH = db.DATA / "graph.db"
    return db.connect(), graph.connect()


def _co(con, **f):
    cols = ", ".join(f)
    con.execute(f"INSERT INTO companies({cols}) VALUES({', '.join('?' * len(f))})", list(f.values()))
    return con.execute("SELECT id FROM companies WHERE key=?", (f["key"],)).fetchone()[0]


def _card(con, cid, owner, card_days, defend=False):
    card = {"headline": "h", "why_now": [], "customer_goal": "g", "pitch": "p", "contact_role": "Head of IT",
            "opener": "o", "email_subject": "s", "email_body": "Hi there, a short note.", "risk": "r"}
    con.execute("INSERT INTO opps(company_id, product, score, fit, intent, timing, relationship, win_prob, signal_ids, "
                "meta, card, card_at, updated) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (cid, "mss", 90, 1, 1, 1, 1, 0.5, "[]", json.dumps({"owner": owner, "defend": defend}),
                 json.dumps(card), (datetime.now(timezone.utc) - timedelta(days=card_days)).isoformat(), "x"))


def seed(con):
    sp = _co(con, key="singapore post", name="Singapore Post", industry="logistics", in_crm=1, is_customer=1,
             owner="Aisha Tan", contract_end=ahead(270))
    ag = _co(con, key="ministry of testing", name="Ministry of Testing", industry="government")
    quiet = _co(con, key="quiet", name="Quiet Pte Ltd", industry="manufacturing", in_crm=1, owner="Mei Ling Ong")
    peer = _co(con, key="peer logistics", name="Peer Logistics", industry="logistics")
    sigs = [(1, sp, "leadership_change", "New Group CIO named", ago(20)),
            (2, sp, "own_breach", "Booking records leaked", ago(10)),
            (3, peer, "own_breach", "Ransomware at Peer Logistics", ago(5))]
    for item, cid, typ, summary, d in sigs:
        con.execute("INSERT INTO signals(item_id, company_id, type, summary, country, date, confidence, url, title) "
                    "VALUES(?,?,?,?,?,?,?,?,?)", (item, cid, typ, summary, "SG", d, 0.9, f"https://news.example/{item}", summary))
    con.execute("INSERT INTO people(company_id, ext_id, name, title, started) VALUES(?,?,?,?,?)",
                (sp, "p1", "Lee Ming", "Group CIO", ago(90)[:7]))
    con.execute("INSERT INTO people(company_id, ext_id, name, title, email, started) VALUES(?,?,?,?,?,?)",
                (sp, "p2", "Joel Chng", "Assistant Vice President, IS Security & Governance", "joel@singpost.com", "2019-04"))
    con.execute("INSERT INTO awards VALUES(?,?,?,?,?,?,?,?,?)",
                ("T-1", "Managed network services", "Ministry of Testing", ago(800), "SINGAPORE TELECOMMUNICATIONS LIMITED",
                 1200000, "connectivity", 36, ahead(240)))
    _card(con, sp, "Aisha Tan", 20, defend=True)
    _card(con, ag, "Ravi Kumar", 0)
    con.commit()
    return {"singpost": sp, "agency": ag, "quiet": quiet}


class Base(unittest.TestCase):
    def setUp(self):
        self.con, self.g = fresh()
        self.ids = seed(self.con)
        self.sp, self.ag, self.quiet = self.ids["singpost"], self.ids["agency"], self.ids["quiet"]

    def row(self, tid):
        return self.con.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()

    def count(self, table):
        return self.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


ENTITIES = [("199201623M", "SINGAPORE POST LIMITED", "Live Company", "1992-03-28", None),
            ("200000001A", "SINGAPORE POST PTE. LTD.", "Struck Off", "2000-01-01", None),
            ("201000002B", "NINJA LOGISTICS PTE. LTD.", "Live Company", "2010-01-01", None),
            ("201500003C", "QUANTIUM SOLUTIONS INTERNATIONAL PTE. LTD.", "Live Company", "2015-01-01",
             "SPEEDPOST INTERNATIONAL PTE. LTD."),
            ("202000004D", "NAMU HOLDINGS PTE. LIMITED", "In Liquidation", "2020-01-01", None),
            ("202100005E", "NAMU HOLDINGS PTE. LTD.", "Live Company", "2021-01-01", None),
            ("201800006F", "BLUE HERON TRADING PTE. LTD.", "Live Company", "2018-01-01", "RED KITE TRADING PTE. LTD."),
            ("201900007G", "RED KITE TRADING PTE. LTD.", "Live Company", "2019-01-01", None)]
LEIS = [("LEI0SINGPOST000000001", "SINGAPORE POST LIMITED", "SG", "SG", "RA000523", "199201623M", "GENERAL", "2012-01-01"),
        ("LEI0QUANTIUM000000002", "QUANTIUM SOLUTIONS INTERNATIONAL PTE. LTD.", "SG", "SG", "RA000523", "201500003C",
         "GENERAL", "2015-02-01"),
        ("LEI0ALPHAFUND00000003", "ALPHA GROWTH FUND VCC", "SG", "SG", "RA000523", "T21VC0001A", "FUND", "2021-05-01"),
        ("LEI0NINJAPARENT000004", "NINJA VAN HOLDINGS LTD", "KY", "SG", "RA000999", "CR-1", "GENERAL", "2014-01-01"),
        ("LEI0NINJALOG000000005", "NINJA LOGISTICS PTE. LTD.", "SG", "SG", "RA000523", "201000002B", "GENERAL", "2016-01-01")]


def seed_graph(g):
    """A tiny graph shaped like the real one: same-name entities, a renamed one, a fund only GLEIF knows, a foreign
    parent, GeBIZ awards under exact and loose supplier names, and news found by tag and by headline."""
    for uen, name, status, reg, former in ENTITIES:
        sp = uen == "199201623M"
        g.execute(f"INSERT INTO entities VALUES({','.join('?' * 14)})",
                  (uen, name, db.key(name), "Local Company", None, status, reg, "53100" if sp else None,
                   "POSTAL ACTIVITIES" if sp else None, None, None, None, former, None))
    for lei, name, country, hq, ra, reg_id, cat, created in LEIS:
        g.execute("INSERT INTO lei VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (lei, name, db.key(name), country, hq, ra, reg_id, cat, "ACTIVE", created))
    g.executemany("INSERT INTO owns VALUES(?,?,?,'PUBLISHED')", [
        ("LEI0QUANTIUM000000002", "LEI0SINGPOST000000001", "IS_DIRECTLY_CONSOLIDATED_BY"),
        ("LEI0QUANTIUM000000002", "LEI0SINGPOST000000001", "IS_ULTIMATELY_CONSOLIDATED_BY"),
        ("LEI0NINJALOG000000005", "LEI0NINJAPARENT000004", "IS_DIRECTLY_CONSOLIDATED_BY")])
    for t in [("T-100", "Mail services", "Ministry of Testing", "2025-03-01", "SINGAPORE POST LIMITED", 250000.0),
              ("T-101", "Parcel lockers", "Ministry of Testing", "2025-06-01", "SINGAPORE POST LIMITED", 90000.0),
              ("T-102", "Courier", "Agency B", "2024-01-01", "NAMU HOLDINGS PTE. LTD.", 50000.0),
              ("T-103", "Trading", "Agency B", "2024-02-01", "NAMU HOLDINGS", 20000.0)]:
        g.execute("INSERT INTO contracts VALUES(?,?,?,?,?,?,?,'Awarded')", (*t[:5], db.key(t[4]), t[5]))
    g.executemany("INSERT INTO news VALUES(?,?,?,?,'eng',?,?,'','',0,?)", [
        ("n1", "2026-11-01T00:00:00", "https://news.example/a", "straitstimes.com", "SN",
         "singapore post;quantium solutions international", "Singapore Post wins parcel deal"),
        ("n2", "2026-11-02T00:00:00", "https://news.example/b", "businesstimes.com.sg", "SN", "",
         "SingPost first-half profit rises"),
        ("n3", "2026-10-01T00:00:00", "https://news.example/c", "thestar.com.my", "MY", "", "SingPost Malaysia note")])
    g.executemany("INSERT INTO mentions VALUES(?,?)", [("singapore post", "n1"), ("quantium solutions international", "n1")])
    g.executemany("INSERT INTO loaded VALUES(?,0,?)", [
        ("acra/a", "2026-11-07T07:44:13"), ("gebiz", "2026-11-07T07:34:42"),
        ("20261107-0000-gleif-goldencopy-lei2-golden-copy.csv.zip", "2026-11-07T07:36:25"),
        ("gdelt/20261108.gkg", "2026-11-08T06:03:53")])
    g.commit()
