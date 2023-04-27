import pandas as pd

# making dataframe
df = pd.read_csv("bankruptcy_data_set.csv")

#number of features (e.g. no. of rows, columns)
#print(df.shape)

#column names and types
#print(df.info())

#remove column Company
df.drop('Company', axis=1, inplace=True)

#remove all "NaN" rows
df.dropna(inplace=True)

#sum WC/TA & RE/TA
df['Sum of WC/TA & RE/TA'] = df['WC/TA'] + df['RE/TA']

#Multiply EBIT/TA & S/TA
df['Product of EBIT/TA & S/TA'] = df['EBIT/TA'] * df['S/TA']


# load dataframe again
df2 = pd.read_csv("bankruptcy_data_set.csv")

# merge combines two data frames with a join operation
# by default an "inner join" on the index is performed, i.e., all instances with the same index are joined
merged_data = df.merge(df2)


# integer based indexing fetches rows, in this case from 10 (inclusive) to 21 (exclusive)
print(merged_data[10:21])

# in general, we can use iloc for integer based locations
print(merged_data.iloc[:, 1:5])

# with lists of strings we get columns
print(merged_data[['WC/TA', 'EBIT/TA']])

# we can also use boolean formulas to filter the rows of data frames
print(merged_data[merged_data['RE/TA'] < -20])

# when we combine multiple conditions, we must use () because we have bitmasks
# we also need to use the bitwise operators & and | instead of the keywords 'and' and 'or'
print(merged_data[(merged_data['RE/TA'] < -20) & (merged_data['Bankrupt'] == 0)])

# we can use the conditions also with loc and also select the columns we want
print(merged_data.loc[(merged_data['RE/TA'] < -20) &
                      (merged_data['Bankrupt'] == 0), ['WC/TA', 'EBIT/TA']])