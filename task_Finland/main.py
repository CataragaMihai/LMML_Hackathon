import unicodedata
from collections import defaultdict

import pandas as pd

# Swedish name -> Finnish name, both already passed through norm() below.
SWEDISH = {
    "helsingfors": "helsinki", "abo": "turku", "tammerfors": "tampere", "esbo": "espoo",
    "vanda": "vantaa", "borga": "porvoo", "villmanstrand": "lappeenranta", "bjorneborg": "pori",
    "raumo": "rauma", "lahtis": "lahti", "tavastehus": "hameenlinna", "uleaborg": "oulu",
    "vasa": "vaasa", "karleby": "kokkola", "jakobstad": "pietarsaari", "nykarleby": "uusikaarlepyy",
    "kristinestad": "kristiinankaupunki", "kasko": "kaskinen", "narpes": "narpio", "malax": "maalahti",
    "korsholm": "mustasaari", "vora": "voyri", "larsmo": "luoto", "kronoby": "kruunupyy",
    "lappo": "lapua", "ilmola": "ilmajoki", "brahestad": "raahe", "tornea": "tornio",
    "idensalmi": "iisalmi", "kajana": "kajaani", "s:t michel": "mikkeli", "s:t karins": "kaarina",
    "nadendal": "naantali", "pargas": "parainen", "nystad": "uusikaupunki", "virmo": "mynamaki",
    "reso": "raisio", "lundo": "lieto", "kimitoon": "kemionsaari", "hango": "hanko",
    "raseborg": "raasepori", "lojo": "lohja", "inga": "inkoo", "sjundea": "siuntio",
    "kyrkslatt": "kirkkonummi", "grankulla": "kauniainen", "sibbo": "sipoo", "kervo": "kerava",
    "tusby": "tuusula", "traskanda": "jarvenpaa", "hyvinge": "hyvinkaa", "vichtis": "vihti",
    "hogfors": "karkkila", "fredrikshamn": "hamina", "ikalis": "ikaalinen", "kumo": "kokemaki",
    "ulvsby": "ulvila", "birkala": "pirkkala", "alavo": "alavus", "keuru": "keuruu", "lovisa": "loviisa",
}


def norm(name):
    """'  Hämeenlinna. ' -> 'hameenlinna': trim, drop trailing dot, lowercase, strip accents."""
    s = name.strip().rstrip(".").strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return SWEDISH.get(s, s)


df = pd.read_csv("routes.csv").drop_duplicates()  # 32 rows logged twice under the same log_id
a, b = df.from_town.map(norm), df.to_town.map(norm)

# Undirected, one edge per town pair whatever the route type; a town is not its own neighbour.
edges = {tuple(sorted(p)) for p in zip(a, b) if p[0] != p[1]}
neighbours = defaultdict(set)
for u, v in edges:
    neighbours[u].add(v)
    neighbours[v].add(u)

degree = pd.Series({t: len(n) for t, n in neighbours.items()}).sort_values(ascending=False)
print(f"towns={len(degree)} connections={len(edges)} (self-check: 116 / 448)")
print(degree.head(10).to_string())
top = degree.head(8)
assert top.is_unique, "self-check says the top 8 degrees are all different"
print("FLAG: SIGMOID_" + "".join(t[0] for t in degree.index[:7]).upper())
