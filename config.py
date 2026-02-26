import pandas as pd
df=pd.read_csv("data/Bitcoin Historical Data.csv")
print (df)
df.head() #display the first 5 rows 
df.tail() #display the first 5 rows
df.head(10) #display the first 10 rows 
df.tail(9)#display the last 9 rows of the csv file 
print(df.shape)
df=df.sort_values(by="Price",ascending=True)
print(df.head())


