import pandas as pd

# making dataframe
df = pd.read_csv("bankruptcy_data_set.csv")

#number of features (e.g. no. of rows, columns)
#print(df.shape)

#column names and types
#print(df.info())


#have to make copy of dataframe to avoid errors
df2 = df

#remove column Company
df2.drop('Company', axis=1, inplace=True)

#remove all "NaN" rows
df2.dropna()

#sum WC/TA & RE/TA
df2['Sum of WC/TA & RE/TA'] = df2['WC/TA'] + df2['RE/TA']

#Multiply EBIT/TA & S/TA
df2['Product of EBIT/TA & S/TA'] = df2['EBIT/TA'] * df2['S/TA']


print(df2)



