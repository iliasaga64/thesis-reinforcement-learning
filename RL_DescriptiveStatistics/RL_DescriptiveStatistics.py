import pandas as pd
import numpy as np

coin = ["BCH","DASH","DGB","ETC","ETH","FCT","GNT","LTC","rev_USDT","STR","XEM","XMR","XRP","ZEC"]
ds = pd.DataFrame(index=coin, columns=["Min", "Quartile1", "Mean", "Median", "Quartile3", "Max"])

for i in coin:
    print(i)
    df      = pd.read_csv("C:\\Users\\Ilyas Agakishiev\\Desktop\\Database\\" + i + ".csv")
    df.date = df.date.astype(int)
    df      = df[df.date % (900*48) == 0]
    df      = df.sort_values(by = "date")
    df.date = pd.to_datetime(df.date, unit = 's')
    df      = df[df.date >= "07/01/2015"]
    df      = df[df.date < "11/01/2018"]
    df["Returns"] = 0
    for j in range(30,len(df.iloc[:,0])):
        df.Returns.iloc[j] = (df.close.iloc[j] - df.close.iloc[j-30]) / df.close.iloc[j-30] * 100
    df      = df.iloc[30:,:]    
    df      = df.sort_values(by = "Returns")

    ds.Min[i] = min(df.Returns.values)   
    ds.Quartile1[i] = df.Returns.iloc[round(len(df.Returns.values)/4)]
    ds.Mean[i] = np.mean(df.Returns.values)
    ds.Median[i] = np.median(df.Returns.values)
    ds.Quartile3[i] = df.Returns.iloc[round(len(df.Returns.values)*3/4)]
    ds.Max[i] = max(df.Returns.values)
        
print(ds)
