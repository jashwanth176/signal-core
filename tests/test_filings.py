import unittest
from datetime import date
from unittest import mock

from agents import llm
from backend import db, filings
from tests import helpers

TODAY = date(2026, 11, 11)


def filing(exchange, company, headline, day="2026-11-10", n=1):
    return filings.item(exchange, "IN", company, headline, "Incorporation of a subsidiary in Singapore", "",
                        f"https://{exchange}.example/{n}", day, ["singapore"])


class GateTest(unittest.TestCase):
    def test_needs_a_place_and_an_action(self):
        self.assertEqual(filings.hits("nse", "Updates. Incorporation of a wholly owned subsidiary in Singapore"),
                         ["singapore", "incorporat", "subsidiar"])
        self.assertEqual(filings.hits("nse", "Incorporation of a wholly owned subsidiary in Dubai"), [])
        self.assertEqual(filings.hits("nse", "Board meeting held in Singapore"), [])

    def test_routine_notices_are_vetoed(self):
        self.assertEqual(filings.hits("nse", "Analyst meet in Singapore on the acquisition"), [])
        self.assertEqual(filings.hits("sgx", "Disposal of the subsidiary in India"), [])
        self.assertEqual(filings.hits("asx", "Becoming a substantial holder: Temasek, Singapore, acquisition"), [])

    def test_action_word_in_the_filers_own_name_does_not_count(self):
        text = "Acme Investments Limited: board to meet in Singapore"
        self.assertTrue(filings.hits("nse", text))
        self.assertEqual(filings.hits("nse", text, "Acme Investments Limited"), [])

    def test_sgx_reads_both_corridors(self):
        self.assertTrue(filings.hits("sgx", "Incorporation of subsidiary. A new plant in Chennai"))
        self.assertTrue(filings.hits("sgx", "Acquisition of Foo Pty Ltd"))
        self.assertEqual(filings.hits("sgx", "The amount was paid in Sydney"), [])  # "amount" is not an MoU

    def test_asx_reads_explorers_only_when_the_headline_names_a_geography(self):
        self.assertTrue(filings.asx_candidate("Expansion into Singapore", "Materials"))
        self.assertFalse(filings.asx_candidate("New facility", "Energy"))
        self.assertTrue(filings.asx_candidate("New facility", "Software & Services"))
        self.assertFalse(filings.asx_candidate("Appendix 3Y, Singapore", "Software & Services"))


class ItemTest(unittest.TestCase):
    def test_title_carries_the_prefix_the_tagger_prompt_names(self):
        it = filing("nse", "Tata  Communications Limited", "Updates")
        self.assertEqual(it["title"], "Stock exchange filing (NSE, 2026-11-10): Tata Communications Limited: Updates")
        self.assertIn(f'Items titled "{filings.PREFIX}"', llm.TAG_SYSTEM)

    def test_body_is_the_summary_then_the_document_around_the_match(self):
        doc = "Letterhead " * 80 + "we have incorporated Acme Pte. Ltd. in Singapore to serve ASEAN customers."
        it = filings.item("nse", "IN", "Acme", "Updates", "Acme has informed the Exchange", doc, "u", "2026-11-10",
                          ["singapore"])
        self.assertTrue(it["body"].startswith("Acme has informed the Exchange "))
        self.assertIn("incorporated Acme Pte. Ltd. in Singapore", it["body"])
        self.assertLess(len(it["body"]), 1400)


class NseTest(unittest.TestCase):
    ROWS = [{"symbol": "ACME", "sm_name": "Acme Limited", "desc": "Updates", "attchmntFile": "-", "seq_id": "1",
             "sort_date": "2026-11-10 18:00:00", "attchmntText": "Incorporation of a subsidiary in Singapore"},
            {"symbol": "ACME", "sm_name": "Acme Limited", "desc": "Updates", "attchmntFile": "-", "seq_id": "2",
             "sort_date": "2026-11-10 19:00:00", "attchmntText": "Acquisition of a stake in Beta Pte. Ltd."},
            {"symbol": "ACME", "sm_name": "Acme Limited", "desc": "Trading Window", "attchmntFile": "-", "seq_id": "3",
             "sort_date": "2026-11-10 20:00:00", "attchmntText": "Closure of trading window"}]

    def test_filings_without_a_pdf_keep_their_own_url(self):
        """NSE writes "-" for no attachment: two such filings by one company once shared a URL and one was lost."""
        with mock.patch.object(filings, "_session"), \
                mock.patch.object(filings, "_get", return_value=mock.Mock(json=lambda: self.ROWS)):
            got = list(filings.nse(date(2026, 11, 9), TODAY))
        self.assertEqual([i["url"][-6:] for i in got], ["#seq=1", "#seq=2"])
        self.assertEqual({i["published"] for i in got}, {"2026-11-10"})


class BseTest(unittest.TestCase):
    def page(self, subject, total=120):
        row = {"SLONGNAME": "Acme Ltd", "NEWSSUB": f"Acme Ltd - 500001 - {subject}", "HEADLINE": subject,
               "NEWS_DT": "2026-11-10T18:00:00", "NEWSID": subject}
        return mock.Mock(json=lambda: {"Table": [row], "Table1": [{"ROWCNT": total}]})

    def run_with(self, *answers):
        with mock.patch.object(filings, "_session"), mock.patch.object(filings.time, "sleep"), \
                mock.patch.object(filings, "_get", side_effect=answers if len(answers) > 1 else answers[0]):
            return [i["title"].split(": ", 2)[2] for i in filings.bse(date(2026, 11, 10), date(2026, 11, 10))]

    def test_a_bad_answer_is_asked_again(self):
        """BSE answers the odd request with a plain "Bad Request"; the same page usually works a moment later."""
        got = self.run_with(self.page("Subsidiary in Singapore"), ValueError("Bad Request"),
                            self.page("Acquisition of Beta Pte. Ltd."), self.page("Nothing more", total=100))
        self.assertEqual(got, ["Subsidiary in Singapore", "Acquisition of Beta Pte. Ltd."])

    def test_a_page_that_stays_bad_is_asked_again_at_the_end(self):
        bad = ValueError("Bad Request")
        got = self.run_with(self.page("Subsidiary in Singapore"), bad, bad, bad, self.page("Branch in Singapore"),
                            self.page("Acquisition of Beta Pte. Ltd."))
        self.assertEqual(got, ["Subsidiary in Singapore", "Branch in Singapore", "Acquisition of Beta Pte. Ltd."])

    def test_a_blocked_exchange_fails_as_a_whole(self):
        with self.assertRaises(ValueError):
            self.run_with(ValueError("Access Denied"))


class CollectTest(unittest.TestCase):
    def setUp(self):
        self.con = helpers.fresh()[0]
        self.asked = {}

    def run_with(self, **found):
        def source(name):
            def fetch(start, end):
                self.asked[name] = start
                if isinstance(found.get(name), Exception):
                    raise found[name]
                return iter(found.get(name, []))
            return fetch
        with mock.patch.object(filings, "EXCHANGES", {n: source(n) for n in ("nse", "bse", "sgx", "asx")}):
            return filings.collect(self.con, today=TODAY)

    def titles(self):
        return [r["title"] for r in self.con.execute("SELECT title FROM items ORDER BY id")]

    def test_saves_items_the_tagger_will_read(self):
        self.assertEqual(self.run_with(nse=[filing("nse", "Acme Limited", "Updates")]), 1)
        row = self.con.execute("SELECT * FROM items").fetchone()
        self.assertEqual((row["source"], row["country"], row["lang"], row["published"], row["tagged"]),
                         ("nse", "IN", "en", "2026-11-10", 0))

    def test_same_disclosure_at_both_indian_exchanges_counts_once(self):
        n = self.run_with(nse=[filing("nse", "Acme Limited", "Updates")],
                          bse=[filing("bse", "Acme Ltd", "Regulation 30", n=2),
                               filing("bse", "Bharat Only Ltd", "Regulation 30", n=3)])
        self.assertEqual(n, 2)
        self.assertEqual([t.split(": ")[1] for t in self.titles()], ["Acme Limited", "Bharat Only Ltd"])
        # also when BSE catches up on a later run, after NSE's copy is already saved
        self.assertEqual(self.run_with(bse=[filing("bse", "Acme Ltd", "Regulation 30", n=4)]), 0)

    def test_two_filings_under_one_category_label_are_both_kept(self):
        n = self.run_with(nse=[filing("nse", "Acme Limited", "Updates", "2026-11-09"),
                               filing("nse", "Acme Limited", "Updates", "2026-11-10", n=2)])
        self.assertEqual(n, 2)

    def test_a_blocked_exchange_is_skipped_and_retried_from_the_same_day(self):
        with mock.patch("builtins.print") as said:
            n = self.run_with(nse=[filing("nse", "Acme Limited", "Updates")], sgx=RuntimeError("403 Access Denied"))
        self.assertEqual(n, 1)
        self.assertIn("skip sgx", said.call_args[0][0])
        self.assertEqual(db.kv_get(self.con, "filings_done"), {k: "2026-11-11" for k in ("nse", "bse", "asx")})
        self.run_with()
        self.assertEqual(self.asked, {"nse": date(2026, 11, 10), "bse": date(2026, 11, 10), "asx": date(2026, 11, 10),
                                      "sgx": date(2026, 10, 28)})

    def test_never_looks_back_more_than_30_days(self):
        db.kv_set(self.con, "filings_done", {"nse": "2026-06-01"})
        self.run_with()
        self.assertEqual(self.asked["nse"], date(2026, 10, 12))
