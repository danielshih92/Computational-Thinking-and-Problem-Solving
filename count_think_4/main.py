# class DisjointSet:
#     def __init__(self, n):
#         self.parent = list(range(n))
#         self.rank = [0] * n
#
#     def find(self, u):
#         if self.parent[u] != u:
#             self.parent[u] = self.find(self.parent[u])  # Path compression
#         return self.parent[u]
#
#     def union(self, u, v):
#         root_u = self.find(u)
#         root_v = self.find(v)
#         if root_u == root_v:
#             return False
#         if self.rank[root_u] < self.rank[root_v]:
#             self.parent[root_u] = root_v
#         elif self.rank[root_u] > self.rank[root_v]:
#             self.parent[root_v] = root_u
#         else:
#             self.parent[root_v] = root_u
#             self.rank[root_u] += 1
#         return True
#
# def tie_breaking_rule(edge1, edge2):
#     return edge1 < edge2
#
# def boruvka_mst(vertices, edges):
#     forest = DisjointSet(len(vertices))
#     mst_edges = []
#     total_weight = 0
#
#     while True:
#         cheapest = [None] * len(vertices)
#
#         for weight, u, v in edges:
#             root_u = forest.find(u)
#             root_v = forest.find(v)
#             if root_u != root_v:
#                 if cheapest[root_u] is None or (weight < cheapest[root_u][0] or
#                                                 (weight == cheapest[root_u][0] and tie_breaking_rule((weight, u, v), cheapest[root_u]))):
#                     cheapest[root_u] = (weight, u, v)
#                 if cheapest[root_v] is None or (weight < cheapest[root_v][0] or
#                                                 (weight == cheapest[root_v][0] and tie_breaking_rule((weight, u, v), cheapest[root_v]))):
#                     cheapest[root_v] = (weight, u, v)
#
#         # Add the cheapest edges found to the MST
#         added_edges = False
#         for edge_info in cheapest:
#             if edge_info is not None:
#                 weight, u, v = edge_info
#                 if forest.union(u, v):
#                     mst_edges.append((u, v))
#                     total_weight += weight
#                     added_edges = True
#
#         if not added_edges:
#             break
#
#     return mst_edges, total_weight
#
# # Example usage:
# vertices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# edges = [(4, 0, 1), (7, 0, 6), (11, 1, 6), (1, 6, 7), (20, 1, 7),(9, 1, 2),(2, 2, 4), (1, 4, 7), (6, 2, 3), (10, 3, 4), (3, 7, 8), (5, 4, 8), (15, 4, 5), (5, 3, 5),(12, 5, 8)]
#
# mst_edges, total_weight = boruvka_mst(vertices, edges)
# print("MST Edges:", mst_edges)
# print("Total Weight:", total_weight)
import sympy as sp

# Define the symbols
fr, er = sp.symbols('fr er')
c = 3 * 10**8  # Speed of light in m/s
h = 8 / 10000  # Height in meters

# Given equations
W = c / (2 * fr * sp.sqrt((er + 1) / 2))  # Width of patch (W)
Ee = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 * (h / W))**(-0.5)  # Effective dielectric constant (Ee)
delta_l = (0.412 * (Ee + 0.3) * (er + 0.264) / ((Ee - 0.258) * (er + 0.8)) * h )*1000 # Length of extension (delta_l)
Leff = (c / (2 * fr * sp.sqrt(Ee)))*1000  # Effective length (Leff)
L = (Leff - 2 * delta_l)*1000  # Actual length of patch (L)
Lg = (6 * h + L )*1000 # Length of ground (Lg)
Wg = (6 * h + W )*1000 # Width of ground (Wg)

# Create a function that evaluates the expressions given fr and er
def calculate_patch_dimensions(fr_value, er_value):
    # Substitute fr and er with the provided values and evaluate the expressions
    W_value = W.subs({fr: fr_value, er: er_value}).evalf()
    Ee_value = Ee.subs({fr: fr_value, er: er_value, W: W_value}).evalf()
    delta_l_value = delta_l.subs({fr: fr_value, er: er_value, Ee: Ee_value}).evalf()
    Leff_value = Leff.subs({fr: fr_value, er: er_value, Ee: Ee_value}).evalf()
    L_value = L.subs({fr: fr_value, er: er_value, Ee: Ee_value, delta_l: delta_l_value}).evalf()
    Lg_value = Lg.subs({fr: fr_value, er: er_value, L: L_value}).evalf()
    Wg_value = Wg.subs({fr: fr_value, er: er_value, W: W_value}).evalf()
    print(W_value)
    print(Ee_value)
    print(delta_l_value)
    print(Leff_value)
    print(L_value)
    print(Lg_value)
    print(Wg_value)
    return {
        'W': W_value,
        'Ee': Ee_value,
        'delta_l': delta_l_value,
        'Leff': Leff_value,
        'L': L_value,
        'Lg': Lg_value,
        'Wg': Wg_value
    }


# Example values for fr (in Hz) and er
example_fr = 2.4*(10**9)  # 1 GHz
example_er = 4.4  # Relative permittivity

# Calculate the dimensions for the example values
calculate_patch_dimensions(example_fr, example_er)








