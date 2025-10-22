"""Třída reprezentující graf"""

from collections import deque
from dataclasses import dataclass, field
from .node import Node
from .edge import Edge

@dataclass
class Graph:
    name: str | None = None
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    is_directed: bool = False
    is_edge_weighted: bool = False
    is_node_weighted: bool = False

    # ===================| Tvorba a hledání v grafu |===================

    # Přidá uzel do grafu
    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    # Přidá hranu do grafu
    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    # Vrátí uzel podle názvu
    def find_node(self, name: str) -> Node | None:
        return next((node for node in self.nodes if node.name == name), None)

    # Vrátí hranu podle názvu
    def find_edge(self, name) -> Edge:
        return next((edge for edge in self.edges if edge.name == name), None)

    # Vyhledá hrany podle vstupního a výstupního uzlu (None = wildcard)
    def find_edges_by_nodes(self, source: Node | None = None, target: Node | None = None) -> list[Edge]:
        def match(e: Edge) -> bool:
            if self.is_directed:
                return (source is None or e.source == source) and (target is None or e.target == target)
            else:
                nodes = {e.source, e.target}
                return (source is None or source in nodes) and (target is None or target in nodes)
        return [e for e in self.edges if match(e)]

    # Vrátí smyčky
    def get_loops(self) -> list[Edge]:
        return [edge for edge in self.edges if edge.source == edge.target]

    # Vrátí násobné hrany
    def get_multi_edges(self) -> list[tuple[Edge, Edge]]:
        multi_edges = []
        for i, x in enumerate(self.edges):
            for j in range(i + 1, len(self.edges)):  # jen každou dvojici jednou
                y = self.edges[j]

                if self.is_directed:
                    if x.source == y.source and x.target == y.target:
                        multi_edges.append((x, y))
                else:
                    ends_x = {x.source, x.target}
                    ends_y = {y.source, y.target}
                    if ends_x == ends_y:
                        multi_edges.append((x, y))

        return multi_edges


    # ===================| Výpis grafu |===================

    # Vypíše graf a jeho vlastnosti
    def __str__(self):
        def shorten_list(items, limit: int):
            if len(items) > limit:  # x položek + poslední
                return "[" + ", ".join(map(str, items[:limit])) + ", ..., " + str(items[-1]) + "]"
            else:
                return "[" + ", ".join(map(str, items)) + "]"

        def yes_no(value: bool) -> str:
            return "✅ Ano" if value else "❌ Ne"

        nodes_str = shorten_list(sorted(self.nodes, key=lambda n: n.name), 15)
        edges_str = shorten_list(self.edges, 3)
        name = f" ({self.name})" if self.name else ""

        output = (
            f"\n\n==========================| GRAF{name} |==========================\n"
            f"Uzly  ({(str(len(self.nodes)) + ')').ljust(8)}{nodes_str}\n"
            f"Hrany ({(str(len(self.edges)) + ')').ljust(8)}{edges_str}\n"
        )

        output += (
            "\n======= VLASTNOSTI GRAFU =======\n"
            f"a. {'Ohodnocený'.ljust(16)} {yes_no(self.is_weighted_graph()).ljust(7)}  | Hrany mají číselnou váhu\n"
            f"b. {'Orientovaný'.ljust(16)} {yes_no(self.is_directed_graph()).ljust(7)}  | Hrany mají určený směr (hrany jsou *uspořádané* dvojice uzlů)\n"
        )

        if self.is_directed:
            output += (
                f"c. {'Slabě souvislý'.ljust(16)} {yes_no(self.is_weakly_connected_graph()).ljust(7)}  | Mezi každými dvěma vrcholy existuje sled, pokud nebereme v potaz orientaci hran (neexistuje žádný nepropojený uzel)\n"
                f"   {'Silně souvislý'.ljust(16)} {yes_no(self.is_strongly_connected_graph()).ljust(7)}  | Mezi každými dvěma vrcholy existuje sled při zachování orientace hran (z každého vrcholu se dostanu do všech ostatních)\n"
            )
        else:
            output += (
                f"c. {'Souvislý'.ljust(16)} {yes_no(self.is_connected_graph()).ljust(7)}  | Mezi každými dvěma vrcholy existuje sled (o libovolné délce)\n"
            )

        output += (
            f"d. {'Prostý'.ljust(16)} {yes_no(self.is_plain_graph()).ljust(7)}  | Neobsahuje násobné hrany\n"
            f"e. {'Jednoduchý'.ljust(16)} {yes_no(self.is_simple_graph()).ljust(7)}  | Neobsahuje násobné hrany (tj. je prostý) a zároveň neobsahuje smyčky\n"
            f"f. {'Rovinný'.ljust(16)} {yes_no(self.is_planar_graph()).ljust(7)}  | Lze nakreslit do roviny tak, že se žádné dvě hrany neprotínají\n"
            f"g. {'Konečný'.ljust(16)} {yes_no(self.is_finite_graph()).ljust(7)}  | Množina uzlů a hran má končený počet prvků\n"
            f"h. {'Úplný'.ljust(16)} {yes_no(self.is_complete_graph()).ljust(7)}  | Každé dva uzly jsou přímo propojeny v obou směrech\n"
            f"i. {'Regulární'.ljust(16)} {yes_no(self.is_regular_graph()).ljust(7)}  | Všechny uzly mají stejný stupeň\n"
            f"j. {'Bipartitní'.ljust(16)} {yes_no(self.is_bipartite_graph()).ljust(7)}  | Množinu uzlů lze rozdělit na dvě disjunktní množiny, uvnitř kterých nejsou žádné dva uzly propojené\n"
        )

        return output

    __repr__ = __str__

    # ===================| Vlastnosti uzlů |===================

    # Následníci uzlu Ug+(u)
    def get_node_successors(self, node: Node) -> list[Node]:
        """Vrací seznam uzlů dostupných z daného uzlu"""
        successors = set()
        for edge in self.edges:
            if self.is_directed:
                if edge.source == node:
                    successors.add(edge.target)
            else:
                if edge.source == node:
                    successors.add(edge.target)
                elif edge.target == node:
                    successors.add(edge.source)
        return sorted(successors, key=lambda n: n.name)

    # Předchůdci uzlu Ug-(u)
    def get_node_predecessors(self, node: Node) -> list[Node]:
        """Vrací seznam uzlů, které mají hranu končící v tomto uzlu"""
        predecessors = set()
        for edge in self.edges:
            if self.is_directed:
                if edge.target == node:
                    predecessors.add(edge.source)
            else:
                if edge.source == node:
                    predecessors.add(edge.target)
                elif edge.target == node:
                    predecessors.add(edge.source)
        return sorted(predecessors, key=lambda n: n.name)

    # Sousední uzly Ug(u)
    def get_node_neighbors(self, node: Node) -> list[Node]:
        """Vrací všechny sousední uzly"""
        neighbors = set()
        for edge in self.edges:
            if edge.source == node:
                neighbors.add(edge.target)
            elif edge.target == node:
                neighbors.add(edge.source)
        return sorted(neighbors, key=lambda n: n.name)

    # Výstupní okolí Hg+(u)
    def get_node_out_neighborhood(self, node: Node) -> set[Edge]:
        """Vrací výstupní hrany uzlu"""
        outgoing = set()
        for edge in self.edges:
            if self.is_directed:
                if edge.source == node:
                    outgoing.add(edge)
            else:
                if edge.source == node or edge.target == node:
                    outgoing.add(edge)
        return outgoing

    # Vstupní okolí Hg-(u)
    def get_node_in_neighborhood(self, node: Node) -> set[Edge]:
        """Vrací vstupní hrany uzlu"""
        incoming = set()
        for edge in self.edges:
            if self.is_directed:
                if edge.target == node:
                    incoming.add(edge)
            else:
                if edge.source == node or edge.target == node:
                    incoming.add(edge)
        return incoming

    # Okolí uzlu Hg(u)
    def get_node_neighborhood(self, node: Node) -> set[Edge]:
        """Vrací všechny hrany spojené s uzlem"""
        if self.is_directed:
            return self.get_node_out_neighborhood(node).union(self.get_node_in_neighborhood(node))
        else:
            return self.get_node_out_neighborhood(node)

    # Výstupní stupeň dg+(u)
    def get_node_out_degree(self, node: Node) -> int:
        """Vrací počet výstupních hran"""
        return len(self.get_node_successors(node))

    # Vstupní stupeň dg-(u)
    def get_node_in_degree(self, node: Node) -> int:
        """Vrací počet vstupních hran"""
        return len(self.get_node_predecessors(node))

    # Stupeň uzlu dg(u)
    def get_node_degree(self, node: Node) -> int:
        """Vrací počet hran spojených s uzlem (smyčka = 2)"""
        if self.is_directed:
            return self.get_node_in_degree(node) + self.get_node_out_degree(node)
        else:
            degree = 0
            for edge in self.edges:
                if edge.source == node or edge.target == node:
                    degree += 1
                    if edge.source == edge.target == node:
                        degree += 1  # Smyčka = 2
            return degree

    # Vypíše všechny vlastnosti uzlu
    def print_node_characteristic(self, name: str):
        node = self.find_node(name)

        if node not in self.nodes:
            print(f"⚠️ Uzel '{name}' v tomto grafu neexistuje!")
            return

        successors = self.get_node_successors(node)
        predecessors = self.get_node_predecessors(node)
        neighbors = self.get_node_neighbors(node)
        out_neighborhood = self.get_node_out_neighborhood(node)
        in_neighborhood = self.get_node_in_neighborhood(node)
        neighborhood = self.get_node_neighborhood(node)
        out_degree = self.get_node_out_degree(node)
        in_degree = self.get_node_in_degree(node)
        degree = self.get_node_degree(node)

        print(f"====== VLASTNOSTI UZLU ({node}) ======")
        print(f"{'1. Následníci uzlu Ug+':<30}= ({len(successors)}) {successors}")
        print(f"{'2. Předchůdci uzlu Ug-':<30}= ({len(predecessors)}) {predecessors}")
        print(f"{'3. Sousední uzly Ug':<30}= ({len(neighbors)}) {neighbors}")
        print(f"{'4. Výstupní okolí uzlu Hg+':<30}= ({len(out_neighborhood)}) {out_neighborhood}")
        print(f"{'5. Vstupní okolí uzlu Hg-':<30}= ({len(in_neighborhood)}) {in_neighborhood}")
        print(f"{'6. Okolí uzlu Hg':<30}= ({len(neighborhood)}) {neighborhood}")
        print(f"{'7. Výstupní stupeň uzlu dg+':<30}= {out_degree}")
        print(f"{'8. Vstupní stupeň uzlu dg-':<30}= {in_degree}")
        print(f"{'9. Stupeň uzlu dg':<30}= {degree}\n")


    # ===================| Vlastnosti grafu |===================

    # Ohodnocený graf
    def is_weighted_graph(self) -> bool:
        return self.is_edge_weighted or self.is_node_weighted

    # Orientovaný graf
    def is_directed_graph(self) -> bool:
        return self.is_directed

    # Souvislý graf – neorientovaný
    def is_connected_graph(self) -> bool:
        if self.is_directed:
            return False  # Souvislost má smysl jen u neorientovaných grafů

        if not self.nodes:
            return True  # Prázdný graf je triviálně souvislý

        start = self.nodes[0]
        visited = self._bfs(start, directed=False)
        return len(visited) == len(self.nodes)

    # Slabě souvislý graf – orientovaný
    def is_weakly_connected_graph(self) -> bool:
        if not self.nodes:
            return True

        start = self.nodes[0]
        visited = self._bfs(start, directed=False)
        return len(visited) == len(self.nodes)

    # Silně souvislý graf – orientovaný
    def is_strongly_connected_graph(self) -> bool:
        if not self.is_directed:
            return self.is_weakly_connected_graph()

        for node in self.nodes:
            reachable = self._bfs(node, directed=True)
            if len(reachable) != len(self.nodes):
                return False

        return True

    # Prostý graf
    def is_plain_graph(self) -> bool:
        seen_edges = set()

        for edge in self.edges:
            if self.is_directed:
                key = (edge.source, edge.target)
            else:
                key = tuple(sorted([edge.source, edge.target], key=lambda n: n.name))

            if key in seen_edges:
                return False
            seen_edges.add(key)

        return True

    # Jednoduchý graf
    def is_simple_graph(self) -> bool:
        if not self.is_plain_graph():
            return False

        for edge in self.edges:
            if edge.source == edge.target:
                return False

        return True

    # Rovinný graf
    def is_planar_graph(self) -> bool:
        n = len(self.nodes)
        e = len(self.edges)

        if self.is_directed:
            # Převod na neorientovaný graf bez ohledu na typ hran
            undirected = Graph(
                name=self.name,
                nodes=self.nodes.copy(),
                edges=[Edge(source=e.source, target=e.target) for e in self.edges],
                is_directed=False
            )
            return undirected.is_planar_graph()

        if n < 5:
            return True

        if e > 3 * n - 6:
            return False

        from itertools import combinations

        neighbors = {node: set(self.get_node_neighbors(node)) for node in self.nodes}

        for subset in combinations(self.nodes, 5):
            if all(b in neighbors[a] for a in subset for b in subset if a != b):
                return False

        for subset in combinations(self.nodes, 6):
            nodes6 = list(subset)
            for partA in combinations(nodes6, 3):
                partB = [x for x in nodes6 if x not in partA]
                inner_A = any(b in neighbors[a] for a in partA for b in partA if a != b)
                inner_B = any(b in neighbors[a] for a in partB for b in partB if a != b)
                if inner_A or inner_B:
                    continue
                if all(b in neighbors[a] for a in partA for b in partB):
                    return False

        return True

    # Konečný graf
    def is_finite_graph(self) -> bool:
        return True # Zadaný graf nemůže být nekonečný...

    # Úplný graf
    def is_complete_graph(self) -> bool:
        n = len(self.nodes)

        if n < 2:
            return True

        if not self.is_directed:
            expected = n * (n - 1) // 2
            actual = set()

            for edge in self.edges:
                if edge.source != edge.target:
                    key = tuple(sorted([edge.source, edge.target], key=lambda n: n.name))
                    actual.add(key)

            return len(actual) == expected
        else:
            expected = n * (n - 1)
            actual = set()

            for edge in self.edges:
                if edge.source != edge.target:
                    actual.add((edge.source, edge.target))

            return len(actual) == expected

    # Regulární graf
    def is_regular_graph(self) -> bool:
        if not self.nodes:
            return False

        degrees = [self.get_node_degree(node) for node in self.nodes]
        return all(d == degrees[0] for d in degrees)

    # Bipartitní graf
    def is_bipartite_graph(self) -> bool:
        color = {}

        for node in self.nodes:
            if node not in color:
                color[node] = 0
                queue = deque([node])

                while queue:
                    current = queue.popleft()
                    for neighbor in self.get_node_neighbors(current):
                        if neighbor not in color:
                            color[neighbor] = 1 - color[current]
                            queue.append(neighbor)
                        elif color[neighbor] == color[current]:
                            return False

        return True

    # ===================| Průchod grafem |===================

    # Průchod grafem do šířky
    def _bfs(self, start: Node, directed: bool = True) -> set[Node]:
        visited = {start}
        queue = deque([start])

        while queue:
            node = queue.popleft()

            neighbors = (
                self.get_node_successors(node) if directed
                else self.get_node_neighbors(node)
            )

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return visited
