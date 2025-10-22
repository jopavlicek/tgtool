from collections import deque


class Edge:
    def __init__(self, source: str, target: str, weight: int | None = None, name: str = None):
        self.source = source
        self.target = target
        self.weight = weight
        self.name = name

    def __str__(self):
        w = f" [{self.weight}]" if self.weight is not None else ""
        n = f"{self.name}: " if self.name is not None else ""
        return f"({n}{self.source} > {self.target}{w})"
    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.source, self.target, self.weight, self.name) == (other.source, other.target, other.weight, other.name)

    def __hash__(self):
        return hash((self.source, self.target, self.weight, self.name))


class Graph:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.nodes: set[str] = set()
        self.edges: list[Edge] = []
        self.directed: bool = False
        self.weighted: bool = False
        self.named: bool = False

    def add_node(self, name: str):
        self.nodes.add(name)

    def add_edge(self, src: str, tgt: str, weight: int | None = None, name: str = None):
        self.edges.append(Edge(src, tgt, weight, name))

    def find_edges_by_name(self, name) -> list[Edge]:
        """Vrací všechny hrany se zadaným jménem."""
        return [edge for edge in self.edges if edge.name == name]

    def find_edges_by_nodes(self, source, target) -> list[Edge]:
        """
        Vrací seznam hran podle zadaných uzlů.
        - Pokud je graf orientovaný, zohledňuje směr.
        - Pokud je neorientovaný, bere libovolný směr.
        - Hvězdička '*' slouží jako zástupný symbol (wildcard):
            - source='*' → nebere se v potaz počáteční uzel
            - target='*' → nebere se v potaz koncový uzel
        """

        if source == "*" and target == "*":
            return self.edges.copy()

        if self.directed:
            results = []
            for edge in self.edges:
                if (source == "*" or edge.source == source) and (target == "*" or edge.target == target):
                    results.append(edge)
            return results

        else:
            results = []
            for edge in self.edges:
                if source == "*" and target != "*":
                    # hledej všechny hrany, které končí nebo začínají v 'target'
                    if edge.source == target or edge.target == target:
                        results.append(edge)
                elif target == "*" and source != "*":
                    # hledej všechny hrany, které končí nebo začínají v 'source'
                    if edge.source == source or edge.target == source:
                        results.append(edge)
                else:
                    # hledej hrany mezi těmito dvěma uzly bez ohledu na směr
                    if {edge.source, edge.target} == {source, target}:
                        results.append(edge)
            return results

    def get_loops(self) -> list[Edge]:
        return [edge for edge in self.edges if edge.source == edge.target]

    def get_all_multi_edges(self) -> list[tuple[Edge, Edge]]:
        multi_edges = []
        for i, x in enumerate(self.edges):
            for j in range(i + 1, len(self.edges)):  # jen každou dvojici jednou
                y = self.edges[j]

                if self.directed:
                    if x.source == y.source and x.target == y.target:
                        multi_edges.append((x, y))
                else:
                    ends_x = {x.source, x.target}
                    ends_y = {y.source, y.target}
                    if ends_x == ends_y:
                        multi_edges.append((x, y))

        return multi_edges



    """
    =============== Vlastnosti uzlu ===============
    """

    # Množina následníků uzlu Ug+(u)
    def get_node_successors(self, node: str) -> list[str]:
        """
        Vrací seznam uzlů, které jsou dostupné z daného uzlu přes výstupní hrany
        - orientovaný: uzly propojené výstupní hranou
        - neorientovaný: všechny sousedící uzly
        """
        successors: set[str] = set()

        for edge in self.edges:
            if self.directed:
                if edge.source == node:
                    successors.add(edge.target)
            else:
                if edge.source == node:
                    successors.add(edge.target)
                elif edge.target == node:
                    successors.add(edge.source)

        return sorted(successors)

    # Množina předchůdců uzlu Ug-(u)
    def get_node_predecessors(self, node: str) -> list[str]:
        """
        Vrací seznam uzlů, které mají výstupní hranu končící v tomto uzlu
        - orientovaný: uzly propojené vstupní hranou
        - neorientovaný: všechny sousedící uzly
        """
        predecessors: set[str] = set()

        for edge in self.edges:
            if self.directed:
                if edge.target == node:
                    predecessors.add(edge.source)
            else:
                if edge.source == node:
                    predecessors.add(edge.target)
                elif edge.target == node:
                    predecessors.add(edge.source)

        return sorted(predecessors)

    # Množina sousedících uzlů Ug(u)
    def get_node_neighbors(self, node: str) -> list[str]:
        """Vrací všechny uzly propojené přes hranu s tímto uzlem"""
        neighbors = set()

        for edge in self.edges:
            if edge.source == node:
                neighbors.add(edge.target)
            elif edge.target == node:
                neighbors.add(edge.source)

        return sorted(neighbors)

    # Výstupní okolí uzlu Hg+(u)
    def get_node_out_neighborhood(self, node: str) -> set[Edge]:
        """
        Vrací seznam výstupních hran uzlu
        - orientovaný: výstupní hrany
        - neorientovaný: všechny hrany spojené s tímto uzlem
        """
        outgoing_edges: set[Edge] = set()

        for edge in self.edges:
            if self.directed:
                if edge.source == node:
                    outgoing_edges.add(edge)
            else:
                if edge.source == node:
                    outgoing_edges.add(edge)
                elif edge.target == node:
                    outgoing_edges.add(edge)

        return outgoing_edges

    # Vstupní okolí uzlu Hg-(u)
    def get_node_in_neighborhood(self, node: str) -> set[Edge]:
        """
        Vrací seznam vstupních hran uzlu
        - orientovaný: vstupní hrany
        - neorientovaný: všechny hrany spojené s tímto uzlem
        """
        incoming_edges: set[Edge] = set()

        for edge in self.edges:
            if self.directed:
                if edge.target == node:
                    incoming_edges.add(edge)
            else:
                if edge.source == node:
                    incoming_edges.add(edge)
                elif edge.target == node:
                    incoming_edges.add(edge)

        return incoming_edges

    # Okolí uzlu Hg(u)
    def get_node_neighborhood(self, node: str) -> set[Edge]:
        """Vrací všechny hrany spojené s tímto uzlem"""
        if self.directed:
            out_n = self.get_node_out_neighborhood(node)
            in_n = self.get_node_in_neighborhood(node)
            return out_n.union(in_n)
        else:
            return self.get_node_out_neighborhood(node)

    # Výstupní stupeň uzlu dg+(u)
    def get_node_out_degree(self, node: str) -> int:
        """Vrací počet výstupních hran uzlu"""
        return len(self.get_node_successors(node))

    # Vstupní stupeň uzlu dg-(u)
    def get_node_in_degree(self, node: str) -> int:
        """Vrací počet vstupních hran uzlu"""
        return len(self.get_node_predecessors(node))

    # Stupeň uzlu dg(u)
    def get_node_degree(self, node_name: str) -> int:
        """Vrací počet hran spojených s uzlem"""
        if self.directed:
            return self.get_node_in_degree(node_name) + self.get_node_out_degree(node_name)
        else:
            degree = 0
            for edge in self.edges:
                if edge.source == node_name or edge.target == node_name:
                    degree += 1
                    if edge.source == edge.target == node_name:
                        degree += 1  # Smyčka přidá druhý stupeň u neorientovaného grafu
            return degree

    # Všechny vlastnosti uzlu
    def print_node_characteristic(self, node: str):
        if node not in self.nodes:
            print(f"⚠️ Uzel '{node}' v tomto grafu neexistuje!")
            return

        print(f"===== Vlastnosti uzlu ({node}) =====")
        print(f"1. Následníci uzlu ({node}):", self.get_node_successors(node))
        print(f"2. Předchůdci uzlu ({node}):", self.get_node_predecessors(node))
        print(f"3. Sousední uzly ({node}):", self.get_node_neighbors(node))
        print(f"4. Výstupní okolí uzlu ({node}):", self.get_node_out_neighborhood(node))
        print(f"5. Vstupní okolí uzlu ({node}):", self.get_node_in_neighborhood(node))
        print(f"6. Okolí uzlu ({node}):", self.get_node_neighborhood(node))
        print(f"7. Výstupní stupeň uzlu ({node}):", self.get_node_out_degree(node))
        print(f"8. Vstupní stupeň uzlu ({node}):", self.get_node_in_degree(node))
        print(f"9. Stupeň uzlu ({node}):", self.get_node_degree(node), "\n")

    """
    =============== Vlastnosti grafu ===============
    """

    # ohodnocený, b) orientovaný, c) souvislý, d) prostý, e) jednoduchý, f) rovinný, g) konečný, h) úplný, i) regulární, j) bipartitní

    def _bfs(self, start: str, directed: bool = True) -> set[str]:
        """Pomocná funkce: BFS průchod grafem (do šířky)."""
        visited = {start}
        queue = deque([start])

        while queue:
            node = queue.popleft()

            # Vyber vhodné sousedy podle typu
            if directed:
                neighbors = self.get_node_successors(node)
            else:
                neighbors = self.get_node_neighbors(node)

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return visited

    # Ohodnocený graf
    def is_weighted_graph(self) -> bool:
        return self.weighted

    # Orientovaný graf
    def is_directed_graph(self) -> bool:
        return self.directed

    # Souvislý graf (pro neorientovaný)
    def is_connected_graph(self) -> bool:
        """
        Vrací True, pokud je graf souvislý (pro neorientované grafy).
        - Pokud je graf orientovaný, vrací False (místo toho se používá slabá nebo silná souvislost).
        """
        if self.directed:
            # Souvislost má smysl jen u neorientovaných grafů
            return False

        if not self.nodes:
            return True  # prázdný graf považujeme za souvislý

        start = next(iter(self.nodes))
        visited = self._bfs(start, directed=False)
        return len(visited) == len(self.nodes)

    # Slabě souvislý graf (pro orientovaný)
    def is_weakly_connected_graph(self) -> bool:
        """
        Vrací True, pokud je graf slabě souvislý.
        - U orientovaného grafu: kontroluje souvislost po odstranění orientace hran.
        - U neorientovaného grafu: kontroluje běžnou souvislost.
        """
        if not self.nodes:
            return True  # prázdný graf považujeme za souvislý

        # Začneme z libovolného uzlu
        start = next(iter(self.nodes))
        visited = self._bfs(start, directed=False)  # ignorujeme orientaci

        return len(visited) == len(self.nodes)

    # Silně souvislý graf (pro orientovaný)
    def is_strongly_connected_graph(self) -> bool:
        """
        Vrací True, pokud je graf silně souvislý.
        Tzn. z každého uzlu se dá dostat do každého jiného.
        """
        if not self.directed:
            # Pro neorientovaný graf silná souvislost = běžná souvislost
            return self.is_weakly_connected_graph()

        for node in self.nodes:
            reachable = self._bfs(node, directed=True)
            if len(reachable) != len(self.nodes):
                return False  # z tohoto uzlu nedosáhneme všech

        return True

    # Prostý graf
    def is_plain_graph(self) -> bool:
        """
        Vrací True, pokud je graf prostý:
        - neobsahuje násobné (duplicitní) hrany mezi stejnými uzly
        """
        seen_edges = set()

        for edge in self.edges:
            # U neorientovaného grafu pořadí uzlů nerozhoduje
            if self.directed:
                key = (edge.source, edge.target)
            else:
                key = tuple(sorted([edge.source, edge.target]))

            if key in seen_edges:
                return False  # nalezena násobná hrana
            seen_edges.add(key)

        return True

    # Jednoduchý graf
    def is_simple_graph(self) -> bool:
        """
        Vrací True, pokud je graf jednoduchý:
        - je prostý (bez násobných hran)
        - nemá smyčky (hrany typu u→u)
        """
        if not self.is_plain_graph():
            return False

        # Zkontroluj smyčky
        for edge in self.edges:
            if edge.source == edge.target:
                return False

        return True

    # Rovinný graf
    def is_planar_graph(self) -> bool:
        """
        Vrací True, pokud je graf pravděpodobně rovinný.
        Používá:
        - Eulerovu nerovnost (nutná podmínka)
        - heuristickou detekci K5 a K3,3
        """
        n = len(self.nodes)
        e = len(self.edges)

        # Rovinnost má smysl jen pro neorientované grafy
        if self.directed:
            # Ignoruj orientaci pro účely testu
            undirected = Graph(self.file_path)
            undirected.nodes = self.nodes.copy()
            undirected.edges = [Edge(e.source, e.target) for e in self.edges]
            undirected.directed = False
            return undirected.is_planar_graph()

        # Malé grafy jsou vždy rovinné
        if n < 5:
            return True

        # Eulerova nerovnost
        if e > 3 * n - 6:
            return False

        # --- Heuristická detekce K5 nebo K3,3 ---
        # vytvoř si mapu sousedů
        neighbors = {node: set(self.get_node_neighbors(node)) for node in self.nodes}

        # Detekce K5: pokud existuje 5 uzlů, které jsou všechny navzájem propojené
        from itertools import combinations

        for subset in combinations(self.nodes, 5):
            sub = list(subset)
            fully_connected = all(
                sub[j] in neighbors[sub[i]] for i in range(5) for j in range(5) if i != j
            )
            if fully_connected:
                return False

        # Detekce K3,3: pokud lze rozdělit 6 uzlů na dvě trojice, kde každá trojice
        # je kompletně propojena s druhou trojicí, ale ne uvnitř
        for subset in combinations(self.nodes, 6):
            nodes6 = list(subset)
            for partA in combinations(nodes6, 3):
                partB = [x for x in nodes6 if x not in partA]
                # Zkontroluj, že nejsou vnitřní hrany uvnitř skupin
                inner_edges_A = any(b in neighbors[a] for a in partA for b in partA if a != b)
                inner_edges_B = any(b in neighbors[a] for a in partB for b in partB if a != b)
                if inner_edges_A or inner_edges_B:
                    continue
                # Zkontroluj úplnou propojenost mezi A a B
                complete_AB = all(b in neighbors[a] for a in partA for b in partB)
                if complete_AB:
                    return False

        return True


    # Konečný graf
    def is_finite_graph(self) -> bool:
        """
        Vrací True, pokud je graf konečný:
        - má konečný (tj. omezený) počet uzlů i hran.
        """
        try:
            return len(self.nodes) < float("inf") and len(self.edges) < float("inf")
        except Exception:
            return False

    # Úplný graf
    def is_complete_graph(self) -> bool:
        """
        Vrací True, pokud je graf úplný:
        - každé dva různé uzly jsou propojeny právě jednou hranou
        - pro orientovaný graf musí existovat obě hrany (A->B i B->A)
        """
        nodes = list(self.nodes)
        n = len(nodes)

        if n < 2:
            return True  # graf s 0 nebo 1 uzlem je triviálně úplný

        # Pro neorientovaný graf
        if not self.directed:
            expected_edge_count = n * (n - 1) // 2
            actual_edge_pairs = set()

            for edge in self.edges:
                pair = tuple(sorted([edge.source, edge.target]))
                if edge.source != edge.target:  # smyčky nepočítáme
                    actual_edge_pairs.add(pair)

            # Musí existovat přesně kombinace všech dvojic
            return len(actual_edge_pairs) == expected_edge_count

        # Pro orientovaný graf
        else:
            expected_edge_count = n * (n - 1)
            actual_edge_pairs = set()

            for edge in self.edges:
                if edge.source != edge.target:  # smyčky nepočítáme
                    actual_edge_pairs.add((edge.source, edge.target))

            return len(actual_edge_pairs) == expected_edge_count

    # Regulární graf
    def is_regular_graph(self) -> bool:
        """
        Vrací True, pokud je graf regulární:
        - všechny uzly mají stejný stupeň (dg(u))
        """
        if not self.nodes:
            return False  # prázdný graf

        # Spočítat stupeň každého uzlu
        degrees = [self.get_node_degree(node) for node in self.nodes]

        # Pokud jsou všechny stupně stejné → regulární
        return all(d == degrees[0] for d in degrees)

    # Bipartitní graf
    def is_bipartite_graph(self) -> bool:
        """
        Vrací True, pokud je graf bipartitní (dvojdílný).
        Používá BFS barvení vrcholů.
        """
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
                            return False  # soused má stejnou barvu ⇒ není bipartitní
        return True

    """
    =============== Výpis grafu ===============
    """

    def __str__(self):
        def shorten_list(items, limit: int):
            """Vrátí zkrácený seznam ve formátu [A, B, ..., Z], pokud je příliš dlouhý."""
            if len(items) > limit:  # x položek + poslední
                return "[" + ", ".join(map(str, items[:limit])) + ", ..., " + str(items[-1]) + "]"
            else:
                return "[" + ", ".join(map(str, items)) + "]"

        nodes_str = shorten_list(sorted(self.nodes), 15)
        edges_str = shorten_list(self.edges, 3)

        def yes_no(value: bool):
            return "✅Ano" if value else "❌Ne"

        output = (
            f"\n====================| GRAF ({self.file_path}) |====================\n"
            f"Uzly  ({(str(len(self.nodes)) + ')').ljust(8)}{nodes_str}\n"
            f"Hrany ({(str(len(self.edges)) + ')').ljust(8)}{edges_str}\n"
            f"\na. {'Ohodnocený'.ljust(17)} {yes_no(self.is_weighted_graph()).ljust(7)}  | Hrany mají číselnou váhu\n"
            f"b. {'Orientovaný'.ljust(17)} {yes_no(self.is_directed_graph()).ljust(7)}  | Hrany mají určený směr (hrany jsou *uspořádané* dvojice uzlů)\n"
        )

        if self.directed:
            output += (f"c. {'Slabě souvislý'.ljust(17)} {yes_no(self.is_weakly_connected_graph()).ljust(7)}  | Mezi každými dvěma vrcholy existuje sled, pokud nebereme v potaz orientaci hran (neexistuje žádný nepropojený uzel)\n"
                       f"   {'Silně souvislý'.ljust(17)} {yes_no(self.is_strongly_connected_graph()).ljust(7)}  | Mezi každými dvěma vrcholy existuje sled při zachování orientace hran (z každého vrcholu se dostanu do všech ostatních)\n")
        else:
            output += f"c. {'Souvislý'.ljust(17)} {yes_no(self.is_connected_graph()).ljust(7)}  | Mezi každými dvěma vrcholy existuje sled (o libovolné délce)\n"

        output += (
            f"d. {'Prostý'.ljust(17)} {yes_no(self.is_plain_graph()).ljust(7)}  | Neobsahuje násobné hrany\n"
            f"e. {'Jednoduchý'.ljust(17)} {yes_no(self.is_simple_graph()).ljust(7)}  | Neobsahuje násobné hrany (tj. je prostý) a zároveň neobsahuje smyčky\n"
            f"f. {'Rovinný'.ljust(17)} {yes_no(self.is_planar_graph()).ljust(7)}  | Lze nakreslit do roviny tak, že se žádné dvě hrany neprotínají\n"
            f"g. {'Konečný'.ljust(17)} {yes_no(self.is_finite_graph()).ljust(7)}  | Množina uzlů a hran má končený počet prvků\n"
            f"h. {'Úplný'.ljust(17)} {yes_no(self.is_complete_graph()).ljust(7)}  | Každé dva uzly jsou přímo propojeny v obou směrech\n"
            f"i. {'Regulární'.ljust(17)} {yes_no(self.is_regular_graph()).ljust(7)}  | Všechny uzly mají stejný stupeň\n"
            f"j. {'Bipartitní'.ljust(17)} {yes_no(self.is_bipartite_graph()).ljust(7)}  | Množinu uzlů lze rodělit na dvě disjunktní množiny, uvnitř kterých nejsou žádné dva uzly propojené\n"
            "\n"
        )

        return output





def parse_graph(file_path: str) -> Graph:
    g = Graph(file_path)

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip empty or comment lines

            parts = line.split()

            # Vertex line: u A;
            if parts[0] == "u":
                name = parts[1].replace(";", "")
                g.add_node(name)

            # Edge line examples:
            # h A > B 2 :h3;
            # h J - K 15;
            elif parts[0] == "h":
                src = parts[1]
                arrow = parts[2]
                tgt = parts[3].replace(";", "")

                # Detect direction
                if arrow in (">", "<"):
                    g.directed = True
                    if arrow == "<":
                        src, tgt = tgt, src
                elif arrow == "-":
                    g.directed = False

                # --- Detect weight robustly ---
                weight = None
                for p in parts[4:]:
                    token = p.strip(";")  # remove trailing semicolons
                    if token.lstrip("-").replace(".", "").isdigit():
                        try:
                            weight = int(token)
                        except ValueError:
                            weight = float(token)
                        g.weighted = True
                        break

                # --- Detect edge name ---
                name = None
                for p in parts:
                    if p.startswith(":"):
                        name = p.replace(":", "").replace(";", "")
                        g.named = True
                        break

                g.add_edge(src, tgt, weight, name)

    # Defaults if undetermined
    if g.directed is None:
        g.directed = False
    if g.weighted is None:
        g.weighted = False

    return g




# =======================================================


if __name__ == "__main__":
    for i in range(1, 21):
        g = parse_graph(f"samples/{str(i).zfill(2)}.tg")
        print(g)






    # print(g.get_all_multi_edges())

    # g1 = parse_graph("samples/08.tg")
    # print(g1)

    # g1.print_node_characteristic("F")
    # g1.print_node_characteristic("A")

    # print(g1.find_edges_by_name("h11"))
    # print(g1.get_loops())

    # váhy hran uzlu G
    # print(sum([node.weight for node in g1.get_node_neighborhood("G")]))
    # print(max([node.weight for node in g1.get_node_neighborhood("G")]))
    # print(min([node.weight for node in g1.get_node_neighborhood("G")]))

    # g.print_node_characteristic("F")
    # print(g.get_edges_by_name("h4"))
    # print(g.get_edges_by_nodes("C", "E"))

    # print(
    #     g.find_edges_by_name("h7")[0].weight,
    #     [node.name for node in g.find_edges_by_nodes("*", "F")]
    # )

    # largest: tuple[str, int] | None = None
    # for node in g.nodes:
    #     if largest == None:
    #         largest = (node, g.get_node_degree(node))
    #     else:
    #         current = (node, g.get_node_degree(node))
    #         if current[1] > largest[1]:
    #             largest = current
    #
    # print(largest)