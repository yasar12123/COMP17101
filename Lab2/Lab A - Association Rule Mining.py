import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split


with open('store_data.csv') as f:
    records = []
    for line in f:
        records.append(line.strip().split(','))


# we first need to create a one-hot encoding of our transactions
te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
df = pd.DataFrame(te_ary, columns=te.columns_)


# use support of 0.005 - low threshold may include to many candidates
# careful selection of rules based on other metrics required
# this means that we use a higher confidence
frequent_item_set = apriori(df, min_support=0.005, use_colnames=True)
#print(frequent_item_set)


# use a high confidence to counterbalance the low support threshold
# order by lift to have the "best" rules at the top
association_rule = association_rules(frequent_item_set, metric="confidence",
                                     min_threshold=0.005).sort_values('lift', ascending=False)
#print(association_rule)


# the mean of a one hot encoded column is the percentage that this value occurs
#print(df['mineral water'].mean())





# split the data into two sets with 50% of the data
subset1, subset2 = train_test_split(df, test_size=0.5, random_state=42)

# create frequent item sets
frequent_item_set_subset1 = apriori(pd.DataFrame(
    subset1), min_support=0.005, use_colnames=True)
frequent_item_set_subset2 = apriori(pd.DataFrame(
    subset2), min_support=0.005, use_colnames=True)

# rules for first set
association_rule_subset1 = association_rules(frequent_item_set_subset1, metric="confidence",
                                             min_threshold=0.5).sort_values('lift', ascending=False)

# rules for second set
association_rule_subset2 = association_rules(frequent_item_set_subset2, metric="confidence",
                                             min_threshold=0.5).sort_values('lift', ascending=False)

pd.set_option("display.max.columns", None)
print(association_rule_subset1)
#print(association_rule_subset2)

