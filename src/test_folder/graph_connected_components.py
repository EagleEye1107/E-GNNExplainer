import networkx as nx
import matplotlib.pyplot as plt


# First Component
G = nx.path_graph(3)

# Second Component
H = nx.Graph()
H.add_node(3)
H.add_node(4)
H.add_edge(3,4)

# Third Component
I = nx.Graph()
I.add_node(5)
I.add_node(6)
I.add_edge(5,6)

# The global Graph
F = nx.compose(G,H)
F = nx.compose(F,I)

# drawing in planar layout
nx.draw_planar(F, with_labels = True)
plt.savefig("./notes/graphs/graph_non_connexe.png")

graph_components = []

print(F.nodes())
print(F.edges())
print(list(nx.connected_components(F)))