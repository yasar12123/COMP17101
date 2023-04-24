import pandas as pd

# making dataframe
df = pd.read_csv("bankruptcy_data_set.csv")


print(df)
all_columns = df.columns.T.tolist()

print(df.columns.T.tolist())

for columns in all_columns:
    df.loc[df[columns] == 'NaN']

#df2 = df[df.'Company' != 'Nan']

print(df)