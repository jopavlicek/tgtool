"""Skript na předmět Teorie grafů"""

from src.node import Node
from src.edge import Edge
from src.graph import Graph
from src.parser import Parser

if __name__ == "__main__":
    for i in range(1, 21):
        g = Parser.parse_graph(file_path=f"samples/{str(i).zfill(2)}.tg")
        print(g)

        # for name in [node.name for node in g.nodes]:
        #     g.print_node_characteristic(name)
