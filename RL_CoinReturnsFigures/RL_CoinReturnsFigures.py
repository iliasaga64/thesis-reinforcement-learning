import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.dates as mdates

coin = ["ETC","BCH","DASH","DGB","ETH","FCT","GNT","LTC","rev_USDT","STR","XEM","XMR","XRP","ZEC"]

for i in coin:
    print(i)
    df      = pd.read_csv("C:\\Users\\Ilyas Agakishiev\\Desktop\\Database\\" + i + ".csv")
    df.date = df.date.astype(int)
    df      = df.sort_values(by = "date")
    df      = df[df.date % (900*48) == 0]
    df.date = pd.to_datetime(df.date, unit = 's')
    df      = df[df.date >= "07/01/2015"]
    df      = df[df.date < "11/01/2018"]
    df["Returns"] = 0
    for j in range(30,len(df.iloc[:,0])):
        df.Returns.iloc[j] = (df.close.iloc[j] - df.close.iloc[j-30]) / df.close.iloc[j-30] * 100
    df      = df.iloc[30:,:]    
    fig, ax = pl.subplots(figsize=(6,4))
    pl.plot(df.date, df.Returns, color = "blue", linewidth = 0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    pl.xlabel("Time") 
    pl.ylabel("Returns (%)")
    pl.title(i + "/BTC monthly returns")
    if (i == "rev_USDT"):
        pl.title("USDT/BTC monthly returns")
    fig.savefig(i + "ret.png", bbox_inches = "tight")