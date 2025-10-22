"""Parser pro načtení grafu ze souboru v .tg formátu"""

from .graph import Graph
from .node import Node
from .edge import Edge

class Parser:
    # Převod vstupních dat na graf
    @staticmethod
    def parse_graph(file_path: str) -> Graph:
        graph = Graph(name=file_path)

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                # Uzly: u A [weight];
                if parts[0] == "u":
                    Parser.parse_node(graph, parts)

                # Hrany: h A > B [weight] [:name];
                elif parts[0] == "h":
                    Parser.parse_edge(graph, parts)

        return graph

    # Načtení řádku s uzlem
    @staticmethod
    def parse_node(graph: Graph, parts: list[str]) -> None:
        name = parts[1].rstrip(";")
        weight = None

        if len(parts) > 2:
            token = parts[2].rstrip(";")
            if token.lstrip("-").replace(".", "").isdigit():
                try:
                    weight = int(token)
                except ValueError:
                    weight = float(token)
                graph.is_node_weighted = True

        graph.add_node(Node(name, weight))

    # Načtení řádku z hranou
    @staticmethod
    def parse_edge(graph: Graph, parts: list[str]):
        src = parts[1]
        arrow = parts[2]
        tgt = parts[3].rstrip(";")

        # směr
        if arrow in (">", "<"):
            graph.is_directed = True
            if arrow == "<":
                src, tgt = tgt, src
        elif arrow == "-":
            graph.is_directed = False

        # váha a jméno (nezáleží na pořadí)
        weight = None
        name = None

        for p in parts[4:]:
            token = p.strip(";")
            if token.startswith(":"):
                name = token[1:]
            elif token.lstrip("-").replace(".", "").isdigit():
                try:
                    weight = int(token)
                except ValueError:
                    weight = float(token)
                graph.is_edge_weighted = True

        graph.add_edge(Edge(Node(src), Node(tgt), weight, name))