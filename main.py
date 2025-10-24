"""Skript na předmět Teorie grafů"""

from src.node import Node
from src.edge import Edge
from src.graph import Graph
from src.parser import Parser

def run_samples():
    for i in range(1, 21):
        g = Parser.parse_graph(file_path=f"samples/{str(i).zfill(2)}.tg")
        print(g)

        g.print_node_characteristic("A")

        g.print_adjacency_matrix()
        g.print_signed_adjacency_matrix()
        g.print_incidence_matrix()
        g.print_distance_matrix()


if __name__ == "__main__":
    # run_samples()

    g = Parser.parse_graph(file_path="samples/01.tg")
    print(g)

    g.print_adjacency_matrix()
    g.print_signed_adjacency_matrix()
    g.print_incidence_matrix()
    g.print_distance_matrix()

    # ============= Příklady výpočtů =============

    # # Kolik smyček obsahuje tento graf?
    # r1 = len(g.get_loops())
    # print(r1)
    #
    # # Kolik uzlů v grafu má stupeň 3?
    # r2_degrees = [g.get_node_degree(n) for n in g.nodes]
    # r2 = len([deg for deg in r2_degrees if deg == 3])
    # print(r2)
    #
    # # Kolik hran má tento graf?
    # r3 = len(g.edges)
    # print(r3)
    #
    # # Jaká je suma stupně uzlů?
    # r4 = sum([g.get_node_degree(n) for n in g.nodes])
    # print(r4)
    #
    # # Největší a nejmenší stupeň uzlu?
    # r5 = (0, None)
    # for n in g.nodes:
    #     deg = g.get_node_degree(n)
    #     if deg > r5[0] or r5[1] is None:
    #         r5 = (deg, n)
    # print(r5)
    #
    # # Suma vah vstupních hran uzlu A
    # r5_node = g.find_node("A")
    # r5_neigh = g.get_node_in_neighborhood(r5_node)
    # r5_weights = [e.weight for e in r5_neigh]
    # r5 = sum(r5_weights)
    # print(r5)

