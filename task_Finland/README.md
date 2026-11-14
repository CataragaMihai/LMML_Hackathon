# Finland · Pipes, Leaks, and the Graph Beneath

**Flag:** `SIGMOID_HOTLINE`

## Run
```bash
cd task_Finland && python main.py
```

## The idea
A group-by on the raw log counts spellings, not towns. Fix that and build the graph:

1. **Duplicates:** 32 rows were logged twice (same `log_id`), so they are dropped.
2. **One town, one name:** trim spaces, drop the trailing `.`, lowercase, strip accents (`Hämeenlinna`, `Hameenlinna` and `HÄMEENLINNA.` become the same town), and map Swedish names to Finnish ones (`Helsingfors`→Helsinki, `Åbo`→Turku, …; Finland is bilingual).
3. **One connection per pair:** the edge is undirected (A→B = B→A), counted once whatever the route type, and self-loops (e.g. `Helsinki`–`Helsingfors`) are dropped.
4. **Rank by degree:** the number of distinct neighbours.

Self-check passes: 116 towns, 448 connections, and the top 8 degrees are all different.

| # | Town | Connections |
|---|---|---|
| 1 | **H**elsinki | 39 |
| 2 | **O**rivesi | 38 |
| 3 | **T**ornio | 33 |
| 4 | **L**ieksa | 29 |
| 5 | **I**matra | 27 |
| 6 | **N**okia | 23 |
| 7 | **E**ura | 22 |

The analyst got Orivesi because Helsinki's edges were split across `Helsinki`, `Helsingfors`, `HELSINKI `, ….
