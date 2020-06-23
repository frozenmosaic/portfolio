import shelve
import re
import string
import time
import os
from datetime import date, datetime, timedelta
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import finquant
from finquant.portfolio import build_portfolio


class YFinanceCrawler:
    timeout = 2
    crumb_link = "https://finance.yahoo.com/quote/{0}/history?p={0}"
    crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
    quote_link = "https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1mo&events=history&crumb={crumb}"
    fpath = os.path.join('portfolio-page', 'projects', 'portfolio-management', 'data', 'test.p')

    def __init__(self, tickers, years_back=10):
        self.tickers = tickers
        self.session = requests.Session()
        self.dateto = date.today()
        self.datefrom = date(
            self.dateto.year - years_back, self.dateto.month, self.dateto.day
        )

    def get_crumb(self, ticker):
        response = self.session.get(
            self.crumb_link.format(ticker), timeout=self.timeout
        )
        response.raise_for_status()
        match = re.search(self.crumble_regex, response.text)
        if not match:
            raise ValueError("Could not get crumb from Yahoo Finance")
        else:
            self.crumb = match.group(1)

    def get_quote(self, ticker):
        if not hasattr(self, "crumb") or len(self.session.cookies) == 0:
            self.get_crumb(ticker)
        url = self.quote_link.format(
            quote=ticker,
            dto=int(time.mktime(self.dateto.timetuple())),
            dfrom=int(time.mktime(self.datefrom.timetuple())),
            crumb=self.crumb,
        )
        response = self.session.get(url)
        response.raise_for_status()

        return pd.read_csv(StringIO(response.text), parse_dates=["Date"])

    def getPeriodIndex(self):
        dateto_str = self.dateto.strftime("%x")
        datefrom_str = self.datefrom.strftime("%x")
        date_index = pd.date_range(start=datefrom_str, end=dateto_str, freq="M")

        return date_index

    def priceDf(self):

        fpath = self.fpath
        try:
            master = shelve.open(fpath, writeback=True)

            # if file is not empty
            if "ver" in master:
                current_ver = master["ver"]
                print(current_ver)

                # if master is up-to-date
                if current_ver == date.today():

                    new = dict()
                    query = []
                    read = []

                    # read from master
                    for ticker in self.tickers:

                        # if master contains ticker
                        if ticker in master["data"]:
                            new[ticker] = master["data"][ticker]
                            read.append(ticker)

                        # if master does not contain ticker
                        else:
                            # crawl for data from Yahoo Finance
                            data = self.get_quote(ticker)
                            prices = data["Adj Close"]

                            # update shelve
                            master["data"][ticker] = prices

                            # add to dict
                            new[ticker] = prices

                            query.append(ticker)

                    print("queried: ", query)
                    print("read: ", read)
                    print("used today's version \n")

                # else if current version is outdated, query new data
                else:
                    new_master = dict(ver=date.today())
                    new_master["data"] = dict()
                    new = dict()

                    for ticker in self.tickers:
                        data = self.get_quote(ticker)
                        prices = data["Adj Close"]
                        new_master["data"][ticker] = prices
                        new[ticker] = prices

                    master["ver"] = new_master["ver"]
                    master["data"] = new_master["data"]
                    print("updated today's ver \n")

            # if file is empty
            else:
                new_master = dict(ver=date.today())
                new_master["data"] = dict()
                new = dict()

                for ticker in self.tickers:
                    data = self.get_quote(ticker)
                    prices = data["Adj Close"]
                    new_master["data"][ticker] = prices
                    new[ticker] = prices

                master["ver"] = new_master["ver"]
                master["data"] = new_master["data"]
                print("created new master \n")

            new = pd.DataFrame(data=new)
            date_index = self.getPeriodIndex()
            new.index = date_index

            master.close()

            return new

        except ValueError:
            print("cannot open shelve")


def buildPf(tickers):
    df = YFinanceCrawler(tickers).priceDf()
    pf = build_portfolio(data=df)
    return (df, pf)


def stats(pf):
    vol = pf.comp_volatility()
    sha = pf.comp_sharpe()
    ret = pf.comp_expected_return(freq=12)

    print("volatility = {:f}\b".format(vol))
    print("sharpe = {:f}\b".format(sha))
    print("return = {:f}\b\n".format(ret))


def castFloat(df, col):
    punct = string.punctuation.replace("|", "")  # to use `|` as separator
    transtab = str.maketrans(dict.fromkeys(punct, ""))

    for c in col:
        df[c] = "|".join(df[c].tolist()).translate(transtab).split("|")
    df = df.astype(dict.fromkeys(col, "f"))
    return df


# tkFid = ["AAL", "AAPL", "BAC", "BND", "DAL", "GS", "INTC", "UNH", "VOO", "XAR", "XOM"]
# dfFid, pfFid = buildPf(tkFid)
# stats(pfFid)

tk5 = [
    "SPY",
    "XAR",
    "BND",
    "VTV",
    "VGT",
    "XLF",
    "AAPL",
    "UNH",
    "MA",
    "V",
    "INTC",
    "NKE",
    "MSFT",
]
df5, pf5 = buildPf(tk5)
stats(pf5)
