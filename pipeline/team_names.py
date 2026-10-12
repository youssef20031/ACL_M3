"""
Map FPL API team names to the names used in the graph and historical CSVs.

FPL renames some clubs between seasons (2024-25 "Ipswich", 2026-27 "Ipswich Town");
the graph keeps the first name it saw, so new data is translated on the way in.
"""

FPL_TO_GRAPH = {
    "Ipswich Town": "Ipswich",
}


def graph_team_name(fpl_name: str) -> str:
    return FPL_TO_GRAPH.get(fpl_name, fpl_name)
