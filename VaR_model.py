# -*- coding: UTF-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import math
import empyrical.stats as emp
from scipy.stats import t
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import binom
from scipy.stats import chi2


class Backtest:
    def __init__(self, actual, forecast, alpha):
        self.index = actual.index
        self.actual = actual.values
        self.forecast = forecast.values
        self.alpha = alpha

    def hit_series(self):
        return (self.actual < self.forecast) * 1

    def number_of_hits(self):
        return self.hit_series().sum()

    def hit_rate(self):
        return self.hit_series().mean()

    def expected_hits(self):
        return self.actual.size * self.alpha

    def duration_series(self):
        hit_series = self.hit_series()
        hit_series[0] = 1
        hit_series[-1] = 1
        return np.diff(np.where(hit_series == 1))[0]

    def plot(self, file_name=None):

        # Re-add the time series index
        r = pd.Series(self.actual, index=self.index)
        q = pd.Series(self.forecast, index=self.index)

        sns.set_context("paper")
        sns.set_style("whitegrid", {"font.family": "serif", "font.serif": "Computer Modern Roman", "text.usetex": True})

        # Hits
        ax = r[r <= q].plot(color="red", marker="o", ls="None", figsize=(6, 3.5))
        for h in r[r <= q].index:
            plt.axvline(h, color="black", alpha=0.4, linewidth=1, zorder=0)

        # Positive returns
        r[q < r].plot(ax=ax, color="green", marker="o", ls="None")

        # Negative returns but no hit
        r[(q <= r) & (r <= 0)].plot(ax=ax, color="orange", marker="o", ls="None")

        # VaR
        q.plot(ax=ax, grid=False, color="black", rot=0)

        # Axes
        plt.xlabel("")
        plt.ylabel("Log Return")
        ax.yaxis.grid()

        sns.despine()
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name, bbox_inches="tight")
        plt.close("all")

    def tick_loss(self, return_mean=True):
        loss = (self.alpha - self.hit_series()) * (self.actual - self.forecast)
        if return_mean:
            return loss.mean()
        else:
            return loss

    def smooth_loss(self, delta=25, return_mean=True):
        """Gonzalez-Rivera, Lee and Mishra (2004)"""
        loss = ((self.alpha - (1 + np.exp(delta * (self.actual - self.forecast))) ** -1) * (
                    self.actual - self.forecast))
        if return_mean:
            return loss.mean()
        else:
            return loss

    def quadratic_loss(self, return_mean=True):
        """Lopez (1999); Martens et al. (2009)"""
        loss = (self.hit_series() * (1 + (self.actual - self.forecast) ** 2))
        if return_mean:
            return loss.mean()
        else:
            return loss

    def firm_loss(self, c=1, return_mean=True):
        """Sarma et al. (2003)"""
        loss = (self.hit_series() * (1 + (self.actual - self.forecast) ** 2) - c * (
                    1 - self.hit_series()) * self.forecast)
        if return_mean:
            return loss.mean()
        else:
            return loss


    def dq_bt(self, hit_lags=4, forecast_lags=1):
        """Dynamic Quantile Test (Engle & Manganelli, 2004)"""
        try:
            hits = self.hit_series()
            p, q, n = hit_lags, forecast_lags, hits.size
            pq = max(p, q - 1)
            y = hits[pq:] - self.alpha  # Dependent variable
            x = np.zeros((n - pq, 1 + p + q))
            x[:, 0] = 1  # Constant

            for i in range(p):  # Lagged hits
                x[:, 1 + i] = hits[pq - (i + 1):-(i + 1)]

            for j in range(q):  # Actual + lagged VaR forecast
                if j > 0:
                    x[:, 1 + p + j] = self.forecast[pq - j:-j]
                else:
                    x[:, 1 + p + j] = self.forecast[pq:]

            beta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
            lr_dq = np.dot(beta, np.dot(np.dot(x.T, x), beta)) / (self.alpha * (1 - self.alpha))
            p_dq = 1 - stats.chi2.cdf(lr_dq, 1 + p + q)

        except:
            lr_dq, p_dq = np.nan, np.nan

        return pd.Series([lr_dq, p_dq],
                         index=["Statistic", "p-value"], name="DQ")


def get_daily_rates(tick_data, do_plots=False):
    """
    Daily rates.
    """

    daily_simple_returns = tick_data.pct_change()
    daily_logarithmic_returns = np.log(tick_data) - np.log(tick_data.shift(1))

    return daily_simple_returns, daily_logarithmic_returns


def roll_VAR(returns, cutoff=0.05, window=250):
    return emp.roll(returns, function=emp.value_at_risk, window=window)


def var_w(x, p=0.99, q=0.99):
    x = pd.DataFrame(x)
    dem = 1 - q ** x.shape[0]
    d = np.zeros(x.shape[0])
    for num, i in enumerate(d):
        d[num] = (q ** (x.shape[0] - (num+1)) * (1 - q))/dem
    x["weg"] = d
    x.sort_values(by=0, ascending=False, inplace=True)

    cumulated_prob = 0
    var_idx = -1
    while cumulated_prob < 1-p:
        var_idx += 1
        cumulated_prob += x.iloc[var_idx,1]

    return x.iloc[var_idx,0]


window_size = 250


def ewma_wojaka(x:pd.DataFrame, lambda_ewma = 0.94, alpha = 0.05):
    x["var"] = np.zeros(x.shape[0])
    window_len = window_size
    for i in range(window_len + 1, len(x)):
        window = []
        sig_i_p1 = math.sqrt((x.iloc[i, 1] ** 2) * lambda_ewma +
                             (1 - lambda_ewma) * (x.iloc[i, 0] ** 2))
        for j in range(i - window_len, i):
            vj = x.iloc[j,0]
            sig_i = x.iloc[j,1]
            scenario_val = vj * (sig_i_p1 / sig_i)
            window.append(scenario_val)
        var = pd.DataFrame(window).apply(lambda x:np.quantile(x,alpha))
        x.iloc[i, 2] = var.values
    return x

def add_series(x,y:pd.Series):
    y.replace(np.nan,0, inplace=True)
    for i in x.index:
        x.iloc[i] = max(x.iloc[i] ,y.iloc[i,0])
    return x


def binominal_backtest(failures,  conf = 0.05):
    """
    Binominal backtest. Implementation based on on https://rdrr.io/cran/Dowd/src/R/BinomialBacktest.R
    """
    size = failures.shape[0]
    failures = np.sum(failures)
    if failures >= size * conf:
        return binom.sf(failures - 1,  size , conf)
    return binom.cdf(failures,  size , conf)


def christ(df, alpha = 0.05):
    """Likelihood ratio framework of Christoffersen (1998)"""
    hits = df
    tr = hits[1:] - hits[:-1]  # Sequence to find transitions

    # Transitions: nij denotes state i is followed by state j nij times
    n01, n10 = (tr == 1).sum(), (tr == -1).sum()
    n11, n00 = (hits[1:][tr == 0] == 1).sum(), (hits[1:][tr == 0] == 0).sum()

    # Times in the states
    n0, n1 = n01 + n00, n10 + n11
    n = n0 + n1

    # Probabilities of the transitions from one state to another
    p01, p11 = n01 / (n00 + n01), n11 / (n11 + n10)
    p = n1 / n

    if n1 > 0:
        # Unconditional Coverage
        uc_h0 = n0 * np.log(1 - alpha) + n1 * np.log(alpha)
        uc_h1 = n0 * np.log(1 - p) + n1 * np.log(p)
        uc = -2 * (uc_h0 - uc_h1)

        # Independence
        ind_h0 = (n00 + n01) * np.log(1 - p) + (n01 + n11) * np.log(p)
        ind_h1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11)
        if p11 > 0:
            ind_h1 += n11 * np.log(p11)
        ind = -2 * (ind_h0 - ind_h1)

        # Conditional coverage
        cc = uc + ind

        # Stack results
        df = pd.concat([pd.Series([uc, ind, cc]),
                        pd.Series([1 - stats.chi2.cdf(uc, 1),
                                   1 - stats.chi2.cdf(ind, 1),
                                   1 - stats.chi2.cdf(cc, 2)])], axis=1)
    else:
        df = pd.DataFrame(np.zeros((3, 2))).replace(0, np.nan)

    # Assign names
    df.columns = ["Statistic", "p-value"]
    df.index = ["Unconditional", "Independence", "Conditional"]

    return df

def kupiec_backtest(failures, conf = 0.05):
    """
    Implementation based on https://www.rdocumentation.org/packages/segMGarch/versions/1.2/source.
    """
    observations = failures.shape[0]
    failures = np.sum(failures)

    try:
        lr = -2 * math.log((((1-conf)**(observations - failures))*(conf**failures)) / ((1 - failures / observations)**(observations-failures)*(failures/observations)**failures))
    except ValueError:
        return 0
    return chi2.sf(lr, df = 1)

if __name__ == '__main__':
    df = pd.read_csv("mbk_d.csv")
    window_size = 250
    df['Date'] = df["Data"].astype('datetime64[ns]')
    #df = df.loc[df["Date"]>= pd.to_datetime("2016")]
    df.set_index(["Date"], inplace=True)
    df = df["Zamkniecie"]
    ret_sim, ret_log = get_daily_rates(df)

    ret_log = - ret_log
    ret_log_roll = ret_log.rolling(window_size)
    df_roll = df.rolling(window_size)
    # Statystyki
    df.plot()
    plt.title("Wykres ceny akcji M-banku")
    plt.xticks(rotation=45)
    plt.show()
    ret_log.plot()
    plt.title("Logarytmiczne sotpy zwrotu")
    plt.xticks(rotation=45)
    plt.show()
    # roling stats

    ret_log_roll.mean().plot()
    plt.title("Wykres sredniej dla okna o wilekości 250")
    plt.show()

    ret_log_roll.std().plot()
    plt.title("Wykres odchylanie standardowego  dla okna o wilekości 250")
    plt.show()
    # test autokrelacji

    plot_acf(ret_log>0)
    plt.show()

    # test na stacjonarnośc
    result = adfuller(ret_log.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    print("XD")
    # histogram i testy normalnosci
    ret_log.hist(bins = 40)
    plt.title("Historyczny rokład stop zwrotu")
    plt.show()
    print(stats.normaltest(ret_log))
    # testy normalnosci dla okna
    pval = ret_log_roll.apply(lambda x:stats.normaltest(x)[1])
    pval.plot()
    plt.xticks(rotation=45)
    plt.title("P-value testów normalności na oknie wielkości 250")
    plt.show()
    # #
    #
    print("Var Historyczny\n ---------------\n")
    # VaR Historyczny
    var_99 = ret_log_roll.apply(lambda x:np.quantile(x,0.99))
    var_95 = ret_log_roll.apply(lambda x:np.quantile(x,0.95))
    ret_log.plot(linewidth=0.1 ,label ="Logaritmiczne stopy zwrotu " )
    var_95.plot(label = "95% VaR" )
    var_99.plot(label ="99% VaR")
    plt.title("Model VaR dla okna wielkości 250")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
    #

    var_95_tick = pd.Series(ret_log > var_95)
    christ_95 = var_95_tick.rolling(window_size).apply(lambda x:christ(x).iloc[0, 1])
    kupiec_95 = var_95_tick.rolling(window_size).apply(kupiec_backtest)
    binomial_95 = var_95_tick.rolling(window_size).apply(binominal_backtest)

    print(f"christorfer 95 {sum(christ_95<0.05)/christ_95.shape[0]}")
    print(f"kupic 95 {sum(kupiec_95 < 0.05) / kupiec_95.shape[0]}")
    print(f"binomial 95  {sum(binomial_95 < 0.05) / binomial_95.shape[0]}")

    var_99_tick = pd.Series(ret_log > var_99)
    christ_99 = var_99_tick.rolling(window_size).apply(lambda x: christ(x, 0.01).iloc[0, 1])
    kupiec_99 = var_99_tick.rolling(window_size).apply(lambda x:kupiec_backtest(x, 0.01))
    binomial_99 = var_99_tick.rolling(window_size).apply(lambda x:binominal_backtest(x, 0.01))

    print(f"christorfer 99 {sum(christ_99 < 0.05) / christ_99.shape[0]}")
    print(f"Kupiec 99 {sum(kupiec_99 < 0.05) / kupiec_99.shape[0]}")
    print(f"Wartosci realne 99 {sum(binomial_99 < 0.05) / binomial_99.shape[0]}")


    #
    print("Var z wagami \n ---------------\n")
    # VaR z wagami
    var_99 = ret_log_roll.apply(var_w)
    var_95 = ret_log_roll.apply(lambda x:var_w(x,p=0.95))
    ret_log.plot(linewidth=0.1 )
    var_95.plot()
    var_99.plot()
    plt.title("Model VaR dla okna wielkości 250")
    plt.legend(["Logaritmiczne stopy zwrotu ","95% VaR", "99% VaR"])
    plt.xticks(rotation=45)
    plt.show()

    var_95_tick = pd.Series(ret_log > var_95)
    christ_95 = var_95_tick.rolling(window_size).apply(lambda x: christ(x).iloc[0, 1])
    kupiec_95 = var_95_tick.rolling(window_size).apply(kupiec_backtest)
    binomial_95 = var_95_tick.rolling(window_size).apply(binominal_backtest)

    print(f"christorfer 95 {sum(christ_95 < 0.05) / christ_95.shape[0]}")
    print(f"kupic 95 {sum(kupiec_95 < 0.05) / kupiec_95.shape[0]}")
    print(f"binomial 95  {sum(binomial_95 < 0.05) / binomial_95.shape[0]}")

    var_99_tick = pd.Series(ret_log > var_99)
    christ_99 = var_99_tick.rolling(window_size).apply(lambda x: christ(x, 0.01).iloc[0, 1])
    kupiec_99 = var_99_tick.rolling(window_size).apply(lambda x: kupiec_backtest(x, 0.01))
    binomial_99 = var_99_tick.rolling(window_size).apply(lambda x: binominal_backtest(x, 0.01))

    print(f"christorfer 99 {sum(christ_99 < 0.05) / christ_99.shape[0]}")
    print(f"Kupiec 99 {sum(kupiec_99 < 0.05) / kupiec_99.shape[0]}")
    print(f"Wartosci realne 99 {sum(binomial_99 < 0.05) / binomial_99.shape[0]}")
    print("EWMA \n ---------------\n")
    #EWMA
    pd.Series.ewm(ret_log, span=50).mean().plot(label = "EWMA",  figsize=(10, 5)  )
    ret_log.plot(linewidth=0.1,label = "Log return")
    plt.legend()
    plt.ylim(0.05, -0.05)
    plt.show()

    ewma = ret_log.copy()
    ewma = pd.DataFrame(ewma)
    ewma["sigma"] = ret_log_roll.std()
    ew_95 = ewma_wojaka(ewma,alpha=0.05)
    ew_99 = ewma_wojaka(ewma,alpha=0.01)
    ret_log.plot(linewidth=0.1, label="Log return")
    ew_95.plot(label="EWMA 95")
    ew_99.plot(label = "EWMA 99")
    plt.legend()
    plt.show()

    print("XD")
    var_95_tick = pd.Series(ret_log > ew_95["var"])
    christ_95 = var_95_tick.rolling(window_size).apply(lambda x: christ(x).iloc[0, 1])
    kupiec_95 = var_95_tick.rolling(window_size).apply(kupiec_backtest)
    binomial_95 = var_95_tick.rolling(window_size).apply(binominal_backtest)

    print(f"christorfer 95 {sum(christ_95 < 0.05) / christ_95.shape[0]}")
    print(f"kupic 95 {sum(kupiec_95 < 0.05) / kupiec_95.shape[0]}")
    print(f"christorfer 95  {sum(binomial_95 < 0.05) / binomial_95.shape[0]}")

    var_99_tick = pd.Series(ret_log > ew_99["var"])
    christ_99 = var_99_tick.rolling(window_size).apply(lambda x: christ(x, 0.01).iloc[0, 1])
    kupiec_99 = var_99_tick.rolling(window_size).apply(lambda x: kupiec_backtest(x, 0.01))
    binomial_99 = var_99_tick.rolling(window_size).apply(lambda x: binominal_backtest(x, 0.01))

    print(f"christorfer 99 {sum(christ_99 < 0.05) / christ_99.shape[0]}")
    print(f"Kupiec 99 {sum(kupiec_99 < 0.05) / kupiec_99.shape[0]}")
    print(f"Wartosci realne 99 {sum(binomial_99 < 0.05) / binomial_99.shape[0]}")

    print("Grach \n ---------------\n")
    # GARCH
    ret_log = -ret_log.dropna()
    am = arch_model(ret_log, vol='Garch', p=1, o=0, q=1, dist='Normal')
    res = am.fit()


    forecasts = res.forecast(start="2010-1-1")
    cond_mean = forecasts.mean
    cond_var = forecasts.variance
    q = am.distribution.ppf([0.01, 0.05])

    value_at_risk = -cond_mean.values - np.sqrt(cond_var).values * q[None, :]
    value_at_risk = pd.DataFrame(
        value_at_risk, columns=['1%', '5%'], index=cond_var.index)
    ax = value_at_risk.plot(legend=False, figsize=(10, 5) )
    xl = ax.set_xlim(value_at_risk.index[0], value_at_risk.index[-1])
    rets_2018 = ret_log
    rets_2018.name = 'Mbank Return'
    c = []
    for idx in value_at_risk.index:
        if rets_2018[idx] > -value_at_risk.loc[idx, '5%']:
            c.append('#000000')
        elif rets_2018[idx] < -value_at_risk.loc[idx, '1%']:
            c.append('#BB0000')
        else:
            c.append('#BB00BB')
    c = np.array(c, dtype='object')
    labels = {
        '#BB0000': '1% Exceedence',
        '#BB00BB': '5% Exceedence',
        '#000000': 'No Exceedence'
    }
    markers = {'#BB0000': 'x', '#BB00BB': 's', '#000000': 'o'}
    for color in ['#BB0000', '#BB00BB']:
        sel = c == color
        ax.scatter(
            rets_2018.index[sel],
            -rets_2018.loc[sel],
            marker=markers[color],
            c=c[sel],
            label=labels[color])
    ax.set_title('Var modelowany za pomoca GRACH(1,1)')
    leg = ax.legend(frameon=False, ncol=3)
    plt.show()


    var_95_tick = pd.Series(ret_log > value_at_risk["5%"])
    christ_95 = var_95_tick.rolling(window_size).apply(lambda x: christ(x).iloc[0, 1])
    kupiec_95 = var_95_tick.rolling(window_size).apply(kupiec_backtest)
    binomial_95 = var_95_tick.rolling(window_size).apply(binominal_backtest)

    print(f"christorfer 95 {sum(christ_95 < 0.05) / christ_95.shape[0]}")
    print(f"kupic 95 {sum(kupiec_95 < 0.05) / kupiec_95.shape[0]}")
    print(f"christorfer 95  {sum(binomial_95 < 0.05) / binomial_95.shape[0]}")

    var_99_tick = pd.Series(ret_log > value_at_risk["1%"])
    christ_99 = var_99_tick.rolling(window_size).apply(lambda x: christ(x, 0.01).iloc[0, 1])
    kupiec_99 = var_99_tick.rolling(window_size).apply(lambda x: kupiec_backtest(x, 0.01))
    binomial_99 = var_99_tick.rolling(window_size).apply(lambda x: binominal_backtest(x, 0.01))

    print(f"christorfer 99 {sum(christ_99 < 0.05) / christ_99.shape[0]}")
    print(f"Kupiec 99 {sum(kupiec_99 < 0.05) / kupiec_99.shape[0]}")
    print(f"Wartosci realne 99 {sum(binomial_99 < 0.05) / binomial_99.shape[0]}")

    # Monte Carlo
    print("Montecarlo \n ---------------\n")
    ret_log =  - ret_log
    mu , sd = stats.norm.fit(ret_log.dropna())
    mean_var_95 = pd.Series(np.zeros(ret_log.shape[0]))
    mean_es_95 = pd.Series(np.zeros(ret_log.shape[0]))
    mean_var_99 = pd.Series(np.zeros(ret_log.shape[0]))
    mean_es_99 = pd.Series(np.zeros(ret_log.shape[0]))
    dist = stats.norm(mu, sd)
    for i in range(10):
        print(i)
        vec = pd.DataFrame(dist.rvs(ret_log.shape[0]))
        # 95%
        var_95 = vec.rolling(window_size).apply(lambda x:np.quantile(x, 0.95))
        es_95 = vec.rolling(window_size).apply(lambda x:np.average(x[x > np.quantile(x, 0.95)].dropna()))
        mean_var_95 = add_series(mean_var_95, var_95)
        mean_es_95 = add_series(mean_es_95,es_95)
        # 99%
        var_99 = vec.rolling(window_size).apply(lambda x: np.quantile(x, 0.99))
        es_99 = vec.rolling(window_size).apply(lambda x: np.average(x[x > np.quantile(x, 0.95)].dropna()))
        mean_es_99 = add_series(mean_es_99, es_99)
        mean_var_99 = add_series(mean_var_99, var_99)

    monte_var_95 = mean_var_95
    monte_es_95 = mean_es_95
    monte_es_99 = mean_es_99
    monte_var_99 = mean_var_99


    monte_var_95.index = ret_log.index
    monte_es_95.index = ret_log.index
    monte_var_95.plot(label="VaR 95%")
    monte_es_95.plot(label="ES 95%")
    ret_log.plot(linewidth = 0.1, label = "Empiryczne stopy zwrotu" )
    plt.title("Monte Carlo VaR 95")
    plt.legend()
    plt.show()

    monte_var_99.index = ret_log.index
    monte_es_99.index = ret_log.index
    monte_var_99.plot(label="VaR 99%")
    monte_es_99.plot(label="ES 99%")
    ret_log.plot(linewidth = 0.1, label = "Empiryczne stopy zwrotu" )
    plt.title("Monte Carlo VaR 99")
    plt.legend()
    plt.show()

    var_95_tick = pd.Series(ret_log > monte_var_95)
    christ_95 = var_95_tick.rolling(window_size).apply(lambda x: christ(x).iloc[0, 1])
    kupiec_95 = var_95_tick.rolling(window_size).apply(kupiec_backtest)
    binomial_95 = var_95_tick.rolling(window_size).apply(binominal_backtest)

    print(f"christorfer 95 {sum(christ_95 < 0.05) / christ_95.shape[0]}")
    print(f"kupic 95 {sum(kupiec_95 < 0.05) / kupiec_95.shape[0]}")
    print(f"christorfer 95  {sum(binomial_95 < 0.05) / binomial_95.shape[0]}")

    var_99_tick = pd.Series(ret_log > monte_var_99)
    christ_99 = var_99_tick.rolling(window_size).apply(lambda x: christ(x, 0.01).iloc[0, 1])
    kupiec_99 = var_99_tick.rolling(window_size).apply(lambda x: kupiec_backtest(x, 0.01))
    binomial_99 = var_99_tick.rolling(window_size).apply(lambda x: binominal_backtest(x, 0.01))

    print(f"christorfer 99 {sum(christ_99 < 0.05) / christ_99.shape[0]}")
    print(f"Kupiec 99 {sum(kupiec_99 < 0.05) / kupiec_99.shape[0]}")
    print(f"Wartosci realne 99 {sum(binomial_99 < 0.05) / binomial_99.shape[0]}")



