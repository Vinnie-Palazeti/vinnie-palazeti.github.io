import json
import requests
import pandas as pd

def bls_series(url, series_dict, date_range):
    all_bls = []

    for start, stop in date_range:  
        print(f'pulling range: {start} to {stop}')     
        
        data = json.dumps(
            {
                "seriesid": list(series_dict.keys()),
                "startyear": start,
                "endyear": stop
            }
        )

        p = requests.post(url, headers={'Content-type': 'application/json'}, data=data).json()['Results']['series']
        date_list = [f"{i['year']}-{i['period'][1:]}-01" for i in p[0]['data']]

        cs = []
        for s in p:
            d = [i['value'] for i in s['data']]
            cs.append(d)

        bls = (
            pd.DataFrame(cs)
            .T
            .set_axis(series_dict.values(), axis=1)
            .astype(float)
            .assign(date = pd.to_datetime(date_list))
            .sort_values('date')
            .reset_index(drop=True)
        )

        bls.to_csv(f'data/bls_{stop}.csv')
        all_bls.append(bls)
    return pd.concat(all_bls)
    

def fred_series(series_id, fred_api_key, series_name = None):
    if not series_name:
        series_name = series_id

    url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={fred_api_key}&file_type=json'

    r = requests.get(url).json()
    fred = (
        pd.DataFrame(r['observations'])
        .rename(columns={'value':series_name})
        .assign(date=lambda df: pd.to_datetime(df.date))
        [['date',series_name]]
    )

    fred['date'] = pd.to_datetime(fred.date.dt.strftime('%Y-%m'))
    fred[series_name] = fred[series_name].astype(float)
    fred = fred.groupby('date')[series_name].mean().reset_index()
    return fred


def ma_pct(d, series):
    d[series] = d[series].rolling(window=12, min_periods=1).mean()
    d[f'{series}_pct_change_12'] = d[series].pct_change(periods=12) * 100
    d[f'{series}_pct_change_24'] = d[series].pct_change(periods=24) * 100
    d[f'{series}_pct_change_36'] = d[series].pct_change(periods=36) * 100
    return d