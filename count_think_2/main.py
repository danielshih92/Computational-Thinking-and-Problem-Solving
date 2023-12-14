def greedy(items, budget):
    # Sort items by discount per dollar spent
    items_sorted = sorted(items, key=lambda x: x[2] / x[1], reverse=True)

    selected_items = []
    total_spent = 0
    total_discount = 0

    for item in items_sorted:
        if total_spent + item[1] <= budget:
            total_spent += item[1]
            total_discount += item[2]
            selected_items.append(item)

    return selected_items



def powerset(items, budget):
    from itertools import chain, combinations

    def total_cost(comb):
        return sum(item[1] for item in comb)

    def total_discount(comb):
        return sum(item[2] for item in comb)

    # Generate all possible combinations of items
    all_combinations = chain(*map(lambda x: combinations(items, x), range(0, len(items) + 1)))

    best_combination = []
    best_discount = 0

    for comb in all_combinations:
        if total_cost(comb) <= budget and total_discount(comb) > best_discount:
            best_combination = comb
            best_discount = total_discount(comb)

    return list(best_combination)


def print_results(algorithm_name, selected_items, execution_time):
    total_value = sum(item[1] for item in selected_items)
    total_discount = sum(item[2] for item in selected_items)

    print(f"{algorithm_name}:")
    print(f"Total number of items taken is: {len(selected_items)}")
    for idx, item in enumerate(selected_items, 1):
        print(f"<{item[0]}, price:{item[1]}, original price:{item[3]}, discount price:{item[2]}>")
    print(f"Total value of items taken is {total_value}")
    print(f"Total discount price of items taken is {total_discount:.2f}")
    print(f"Execution time: {execution_time:.6f} seconds\n")


def buildItems():
    names = ['1,紳士鞋', '2,平板保護套', '3,有線耳機', '4, 列表機', '5, 太陽眼鏡', '6, 炭板跑鞋', '7, 慢跑鞋', '8, 運動手錶', '9, 無限喇叭',
             '10, 運動服', '11, 足球運動套裝', '12 運動鞋', '13, 筆電帶', '14, 藍芽耳機', '15, 滑板', '16, 泳衣', '17, 遊戲機', '18, 水族箱', '19, 自行車頭盔',
             '20, 吹風機', '21, 床單', '22, 行動電源']
    prices = [878, 1140, 2280, 760, 1330, 2099, 2508, 2850, 3762, 1539, 3190, 2153, 3724, 2964, 1406, 1214, 1482, 1060,
              1140, 1482, 1209, 1518]
    oriprices = [1406, 1900, 5700, 2280, 3420, 3418, 3230, 3487, 7562, 2052, 3420, 2658, 4331, 4940, 2280, 2052, 2470,
                 1250, 1290, 2280, 1512, 1786]
    discounts = [528, 760, 3420, 1520, 2090, 1319, 722, 637, 3800, 513, 230, 505, 607, 1976, 874, 838, 988, 190, 151,
                 798, 302, 268]

    items = list(zip(names, prices, discounts, oriprices))
    return items


items = buildItems()
budget = 30000
# ... [rest of the code remains unchanged]
import time
# Greedy
start_time = time.time()
selected_greedy = greedy(items, budget)
end_time = time.time()
print_results("Greedy Algorithm", selected_greedy, end_time - start_time)

# Powerset
start_time = time.time()
selected_powerset = powerset(items, budget)
end_time = time.time()
print_results("Optimal Algorithm using Powerset", selected_powerset, end_time - start_time)
