import sys
import pandas as pd
import datetime

def load_data_up_to_date(path, max_date, count):
    df = pd.read_csv(path)
    
    print(df.shape)
    
    df = df[df.apply(lambda x: datetime.datetime.strptime(x.Date, "%Y-%m-%d") < max_date, axis=1)]
    
    print(df.shape)
    
    # Sort values in ascending order by date
    df = df.sort_values(by=['Date'])
    
    print(df.shape)
    df = df.iloc[-count:,]
    print(df.shape)
    
    return df