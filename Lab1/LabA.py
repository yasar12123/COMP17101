import pandas as pd

# making dataframe
df = pd.read_csv("bankruptcy_data_set.csv")

#number of features (e.g. no. of rows, columns)
print(df.shape)

#column names and types
print(df.info())

#remove column Company
df.drop('Company', axis=1, inplace=True)




