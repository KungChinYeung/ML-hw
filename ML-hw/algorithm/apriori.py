from itertools import chain, combinations
from collections import defaultdict

# 生成候选项集
def generate_candidates(itemset, k):
    candidates = set()
    for item1 in itemset:
        for item2 in itemset:
            union_set = item1.union(item2)
            if len(union_set) == k:
                candidates.add(union_set)
    return candidates

# 过滤非频繁项集
def filter_candidates(candidates, itemsets, min_support):
    candidate_counts = defaultdict(int)

    for transaction in itemsets:
        for candidate in candidates:
            if candidate.issubset(transaction):
                candidate_counts[tuple(candidate)] += 1

    frequent_candidates = set()
    for candidate, count in candidate_counts.items():
        support = count / len(itemsets)
        if support >= min_support:
            frequent_candidates.add(candidate)

    return frequent_candidates

# 生成频繁项集
def apriori(itemsets, min_support):
    itemsets = [set(itemset) for itemset in itemsets]
    k = 2
    frequent_itemsets = set()

    while True:
        candidates = generate_candidates(frequent_itemsets, k)
        frequent_candidates = filter_candidates(candidates, itemsets, min_support)

        if not frequent_candidates:
            break

        frequent_itemsets.update(frequent_candidates)
        k += 1

    return frequent_itemsets

if __name__ == '__main__':
    # 示例数据集，每个元素代表一个购物篮
    dataset = [['apple', 'banana', 'orange'],
               ['banana', 'orange', 'grape'],
               ['apple', 'banana', 'grape'],
               ['apple', 'orange', 'grape'],
               ['apple', 'banana']]

    min_support = 0.4
    frequent_itemsets = apriori(dataset, min_support)

    print("Frequent Itemsets:")
    for itemset in frequent_itemsets:
        print(itemset)



