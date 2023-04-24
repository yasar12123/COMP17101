import pandas as pd

# making dataframe
df = pd.read_csv("bankruptcy_data_set.csv")

#output the dataframe
print(df)

#number of instances (e.g. no. of rows)
print(len(df))
#number of features (e.g. no. of rows, columns)
print(df.shape)

#quick read of dataset (first 5 rows)
print(df.head())

#quick read of dataset (last 5 rows)
print(df.tail())

#to view all columns
pd.set_option("display.max.columns", None)
print(df)

#view stats, column and data type
print(df.info())


#basic stats of dataset (default only includes numeric columns)
print(df.describe())
#to view stats of other data types
print(df.describe(include=object))



#remove columns
df.drop('Company', axis=1, inplace=True)