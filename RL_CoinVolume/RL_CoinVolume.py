import pandas as pd

coin = ["ETH", "LTC", "XRP", "rev_USDT", "ETC", "DASH", "XMR", "XEM", "FCT", "GNT", "ZEC"]
#coin = ["rev_USDT", "ETH", "XRP", "STR", "XMR", "ETC", "DASH", "LTC", "BCH", "ZEC", "DGB"]
v = pd.DataFrame(0, index = coin, columns = ["Volume"])

for i in range(11):
    df          = pd.read_csv("C:\\Users\\Ilyas Agakishiev\\Desktop\\Database\\" + coin[i] + ".csv")
    df.date     = df.date.astype(int)
    df          = df.sort_values(by = "date")
    df.date     = pd.to_datetime(df.date, unit = 's')
    df          = df[df.date >= "04/04/2018"]    # df[df.date >= "08/02/2018"]
    df          = df[df.date < "05/04/2018"]     # df[df.date < "09/02/2018"]
    v.iloc[i,0] = pd.DataFrame.sum(df.volume)
    
print(v)