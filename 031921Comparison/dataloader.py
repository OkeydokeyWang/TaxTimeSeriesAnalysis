import sys
import pandas as pd
import datetime
from pandas.tseries.offsets import BDay


def load_data_up_to_date(path, max_date, count, remove_holidays=True):
    df = pd.read_csv(path)
    
    print(df.shape)

    if (remove_holidays):
        isBusinessDay = BDay().is_on_offset
        match_series = pd.to_datetime(df['Date']).map(isBusinessDay)
        df = df[match_series]

    print(df.shape)
    
    df = df[df.apply(lambda x: datetime.datetime.strptime(x.Date, "%Y-%m-%d") < max_date, axis=1)]
    
    print(df.shape)
    
    # Sort values in ascending order by date
    df = df.sort_values(by=['Date'])
    
    print(df.shape)
    df = df.iloc[-count:,]
    print(df.shape)
    
    return df