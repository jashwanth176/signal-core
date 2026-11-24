"""Signal taxonomy, Singtel product catalogue, and the rules table linking them.

Rule weights are priors. Feedback and won/lost outcomes move them (see engine.weights).
"""

SIGNALS = {
    "new_market_expansion": "Opens an office, entity, factory or hub in a NEW country",
    "new_facility": "New site in a country it already operates in: warehouse, plant, campus, port terminal",
    "hiring_tech": "Hiring network, cloud or security engineers in volume, or building a tech team or a new department",
    "data_regulation": "A data protection / localisation / cyber rule now applies to it or its sector",
    "peer_breach": "A company in the same sector or market suffers a cyber attack or outage",
    "own_breach": "The company itself suffers a cyber attack, data leak or major IT outage",
    "merger_acquisition": "Merger, acquisition, divestment or joint venture",
    "tender_published": "Publishes a tender / RFP for network, cloud, security or IoT work",
    "leadership_change": "New CIO, CTO, CISO, CDO or head of IT / infrastructure",
    "contract_ending": "A connectivity / IT contract with a competitor is ending or up for renewal",
    "digital_transformation": "Announces cloud migration, ERP move, AI programme, digital overhaul, or a new service or "
                              "business line that runs on technology",
    "funding_capex": "Raises capital, IPOs, or announces a capex / IT budget increase",
    "iot_rollout": "Fleet tracking, smart warehouse, smart factory, sensors, connected assets",
    "tech_partnership": "Strategic partnership with a tech vendor or hyperscaler",
    "sustainability_ops": "Energy / emissions programme needing monitoring, data or smart-building tech",
}

PRODUCTS = {
    "sdwan": "Managed SD-WAN",
    "intl": "International Connectivity (IP-VPN / IPLC)",
    "mss": "Managed Security & SOC",
    "cloud": "Multi-Cloud & Sovereign Cloud",
    "dc": "Data Centre & Colocation",
    "iot": "IoT & Fleet Connectivity",
    "5g": "5G Private Network & Edge",
}

# signal -> {product: prior weight 0..1}
RULES = {
    "new_market_expansion": {"sdwan": 0.7, "intl": 0.7, "mss": 0.4},
    "new_facility": {"sdwan": 0.5, "5g": 0.5, "iot": 0.4},
    "hiring_tech": {"mss": 0.5, "sdwan": 0.4, "cloud": 0.4},
    "data_regulation": {"cloud": 0.7, "dc": 0.5, "mss": 0.4},
    "peer_breach": {"mss": 0.6},
    "own_breach": {"mss": 0.8},
    "merger_acquisition": {"sdwan": 0.7, "mss": 0.5, "cloud": 0.4},
    "tender_published": {},  # product comes from the tender text (tagger product_hint)
    "leadership_change": {"mss": 0.4, "cloud": 0.4, "sdwan": 0.3},
    "contract_ending": {"sdwan": 0.6, "intl": 0.5, "mss": 0.4},
    "digital_transformation": {"cloud": 0.6, "sdwan": 0.4, "5g": 0.3},
    "funding_capex": {"cloud": 0.4, "sdwan": 0.3},
    "iot_rollout": {"iot": 0.8, "5g": 0.4},
    "tech_partnership": {"cloud": 0.3},
    "sustainability_ops": {"iot": 0.5},
}

# Who to talk to, per product
CONTACT_ROLE = {
    "sdwan": "Regional Head of IT Infrastructure",
    "intl": "Head of Network / Regional IT Infrastructure",
    "mss": "Chief Information Security Officer",
    "cloud": "CIO / Head of Cloud Platforms",
    "dc": "Head of Infrastructure & Data Centre",
    "iot": "Head of Operations Technology / Supply Chain",
    "5g": "Head of Operations Technology / Plant Director",
}

INDUSTRY_FIT = {
    "logistics": 1.0, "manufacturing": 1.0, "government": 0.9, "finance": 0.9,
    "healthcare": 0.8, "retail": 0.8, "energy": 0.8, "technology": 0.7,
    "real_estate": 0.6, "other": 0.5,
}
SIZE_FIT = {"large": 1.0, "mid": 0.8, "small": 0.5, "unknown": 0.7}
HOME = "SG"
REGION = {"SG", "MY", "ID", "TH", "VN", "PH", "IN", "AU", "JP", "KR", "HK", "TW", "CN"}
COUNTRIES = {"SG": "Singapore", "MY": "Malaysia", "ID": "Indonesia", "TH": "Thailand", "VN": "Vietnam", "PH": "Philippines",
             "IN": "India", "AU": "Australia", "JP": "Japan", "KR": "South Korea", "HK": "Hong Kong", "TW": "Taiwan",
             "CN": "China", "US": "United States", "GB": "United Kingdom", "DE": "Germany", "FR": "France", "NL": "Netherlands",
             "CH": "Switzerland", "KY": "Cayman Islands", "BM": "Bermuda", "VG": "British Virgin Islands", "LU": "Luxembourg",
             "AE": "UAE", "NZ": "New Zealand", "IE": "Ireland", "ZA": "South Africa", "NG": "Nigeria", "TR": "Turkey",
             "BR": "Brazil", "CA": "Canada", "MX": "Mexico", "IT": "Italy", "ES": "Spain", "BE": "Belgium", "SE": "Sweden",
             "DK": "Denmark", "NO": "Norway", "SA": "Saudi Arabia", "QA": "Qatar", "KH": "Cambodia", "MM": "Myanmar",
             "LK": "Sri Lanka", "BD": "Bangladesh", "PK": "Pakistan", "MO": "Macau", "BN": "Brunei", "LA": "Laos",
             "MU": "Mauritius", "GH": "Ghana", "KE": "Kenya", "EG": "Egypt", "CL": "Chile", "AR": "Argentina", "PL": "Poland"}

SINGTEL_GROUP = r"SINGAPORE TELECOM|SINGTEL|^NCS (PTE|COMMUNICATIONS)"

# Signals whose product comes from the item itself (tagger product_hint / GeBIZ category), with prior weight
HINT_PRIOR = {"tender_published": 0.8, "contract_ending": 0.6}
# When not to call (engine.held). A company-wide cut holds every card for HOLD_MONTHS. A contract just signed with
# another supplier holds that product's cards until RIVAL_BACK months before an assumed RIVAL_TERM-month term ends:
# the same 36 months assumed for a GeBIZ award with no stated term, and the 9 months where Timing peaks.
HOLD_MONTHS, RIVAL_TERM, RIVAL_BACK = 6, 36, 9
# A contract shows on the renewal radar from RADAR_OPEN months before its estimated end. It was 18 until the GeBIZ
# back-test (proof.backtest_gebiz, data to March 2026): 4 of the 19 renewals the Singtel group won were awarded 18 to
# 21 months before the earlier contract's estimated end, so 18 flagged 10 of 19 and 21 flags 14. The window was chosen
# on those same 19 tenders; the share of past flags followed by a similar award is the same for both (34%).
RADAR_OPEN = 21
PLACE_SIGNALS = ("new_market_expansion", "new_facility")  # growth tied to a country: a cut there cancels it
GEBIZ_PRODUCT = {"connectivity": "sdwan", "cloud": "cloud", "cyber": "mss", "iot": "iot"}
PRODUCT_CATEGORY = {"sdwan": "connectivity", "intl": "connectivity", "5g": "connectivity",
                    "cloud": "cloud", "dc": "cloud", "mss": "cyber", "iot": "iot"}

# Owner for accounts not yet in the CRM. One seller uses Signal, so every account is theirs.
# Shortcut: per-industry owners ({"logistics": "...", ...}) if a team ever shares one Signal
TERRITORY = {}
DEFAULT_OWNER = "Aisha Tan"

# Singtel's own arms abroad: the regional associates and Optus, by country. A customer's or prospect's site in one of these
# countries can be reached through them (Abroad and Way in). Source: Singtel's company page lists the associates.
ARMS = {"IN": ("Bharti Airtel", "regional associate"), "ID": ("Telkomsel", "regional associate"),
        "TH": ("AIS", "regional associate"), "PH": ("Globe Telecom", "regional associate"), "AU": ("Optus", "subsidiary")}
ARMS_SOURCE = {"name": "Singtel, Our company", "date": "", "url": "https://www.singtel.com/about-us/company"}

# Never pitch to ourselves, our associates, or competitors
EXCLUDE = {"singtel", "ncs", "optus", "bharti airtel", "airtel", "telkomsel", "ais", "globe telecom",
           "starhub", "m1", "ntt", "tata communications", "simba telecom", "myrepublic", "viettel", "vnpt"}

# Account plays: name -> (label, what it watches for, task kind). The checks live in agents.PLAY_CHECKS.
PLAYS = {
    "renewal_due": ("Renewal due", "A government or CRM contract on the account ends in 6 to 12 months.", "renewal_pack"),
    "new_leader": ("New leader", "A new CIO, CTO, CISO or IT head in the last 90 days, or a known contact new in role "
                   "in the last 180 days.", "welcome"),
    "peer_breach": ("Breach at a peer", "A cyber attack or outage at the account or a company in its industry in the "
                    "last 30 days.", "briefing"),
    "new_entity": ("New Singapore entity", "A new company registered with ACRA in the account's group in the last 90 "
                   "days.", "intro"),
    "abroad_entity": ("New entity abroad", "A company in the account's corporate group registered with GLEIF outside "
                      "Singapore in the last 180 days.", "intro"),
}

# Buying committee role slots: (slot, label, title regex), checked in this order; one person fills one slot
COMMITTEE = [
    ("tech", "Tech leader", r"\b(cio|cto|cdo|chief (information|digital|technology))\b|head of (group )?(it|information technology)\b(?! security)|\b(vp|vice president)\b.*\b(it|information technology)\b"),
    ("security", "Security", r"ciso|security"),
    ("infra", "Infrastructure and network", r"infrastructure|network|it operations"),
    ("digital", "Digital and operations", r"digital|operations technology|innovation"),
    ("procurement", "Procurement", r"procurement|purchasing|sourcing"),
    ("finance", "Finance", r"\bcfo\b|financ"),
]

# Suggested questions on the account page's Ask tab
ASK_CHIPS = ["Who is their telco today?", "Which contracts end in the next 12 months?",
             "What went wrong for them recently?", "Who should I talk to first?"]

# Team rules a manager wants true on every account: name -> (label, the rule in one sentence). Checks: agents.RULE_CHECKS
RULES_TEAM = {
    "no_senior_it": ("No senior IT contact", "A renewal inside 180 days needs a named tech leader, security or infrastructure contact."),
    "single_threaded": ("Single-threaded", "An account we defend needs more than one known contact."),
    "stale_card": ("Card waiting", "A card older than 14 days needs a decision."),
    "unseen": ("Not opened", "An account with new facts in the last 30 days should be opened by its owner."),
}

# Account plan sections, in order: (key, heading)
PLAN_SECTIONS = [("situation", "Situation"), ("goals", "Their goals"), ("position", "Our position"),
                 ("committee", "Buying committee"), ("risks", "Risks"), ("next", "Next steps")]
