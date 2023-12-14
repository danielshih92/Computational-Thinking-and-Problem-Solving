from timeit import default_timer as timer


def knapsack_non_recursive(values, weights, W):
    n = len(values)
    K = [[0 for w in range(W + 1)] for i in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if weights[i - 1] <= w:
                K[i][w] = max(K[i - 1][w], K[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                K[i][w] = K[i - 1][w]

    return K


def knapsack_recursive(values, weights, W):
    n = len(values)
    K = [[-1 for w in range(W + 1)] for i in range(n + 1)]

    def ks(n, w):
        if n == 0 or w == 0:
            return 0
        if K[n][w] != -1:
            return K[n][w]
        if weights[n - 1] <= w:
            K[n][w] = max(ks(n - 1, w), ks(n - 1, w - weights[n - 1]) + values[n - 1])
        else:
            K[n][w] = ks(n - 1, w)
        return K[n][w]

    ks(n, W)
    return K


def select_items(K, weights):
    w = len(K[0]) - 1
    n = len(K) - 1
    items = set()

    while n > 0 and w > 0:
        if K[n][w] != K[n - 1][w]:
            items.add(n)
            w -= weights[n - 1]
        n -= 1

    return items


values = [8, 8, 1, 18, 14, 4, 16, 1]
weights = [3, 2, 7, 19, 3, 10, 1, 14]
W = 20

# Non recursive version
start = timer()
K_non_recursive = knapsack_non_recursive(values, weights, W)
end = timer()
print(values,"<--values")
print(weights,"<--weights")
for row in K_non_recursive:
    print(row)
print(select_items(K_non_recursive, weights)," <--selected items")
print(K_non_recursive[-1][-1]," <-- Total value selected")
print("Execution time of non recursive version is:", end - start)
print("**************************************************************************")

# Recursive version
start = timer()
K_recursive = knapsack_recursive(values, weights, W)
end = timer()
print(values,"<--values")
print(weights,"<--weights")
for row in K_recursive:
    print(row)
print(select_items(K_recursive, weights)," <--selected items")
print(K_recursive[-1][-1]," <-- Total value selected")
print("Execution time of recursive version is:", end - start)



