import os
import networkx as nx
import matplotlib.pyplot as plt

def build_file_graph(root_dir):
    G = nx.DiGraph()  # Directed graph for showing parent-child relationships

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            G.add_edge(dirpath, full_path)

        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            G.add_edge(dirpath, full_path)

    return G


def visualize_file_graph(G):
    pos = nx.spring_layout(G, k=0.1, iterations=100)
    plt.figure(figsize=(15, 15))
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, arrows=True)
    plt.savefig("output.png")




if __name__ == "__main__":
    root_directory = "/DATA/deep_learning/eeg-to-img/"  # Change this to your project folder
    file_graph = build_file_graph(root_directory)
    visualize_file_graph(file_graph)
