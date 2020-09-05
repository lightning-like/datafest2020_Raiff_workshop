"""
A/B total monitoring
get total overview on population
std -
mean -
"""
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def uniq(df):
    """ uniq mane will be in column """
    return df.nunique()


def get_q(q_=0.1):
    """ 1 func for all quantile"""

    def quantile(df):
        """ wrapped  """
        return df.quantile(q_)

    quantile.__name__ = f'{quantile.__name__}_{q_}'
    return quantile


def naming(i: float):
    """short names for buckets """
    try:
        return f'{int(i / 1e3)}K' if i < 1e6 else f'{int(i / 1e6)}KK'
    except OverflowError:
        return f'{i}'


def plot_boot(_boot_data, ci=0.05):
    """
    plot ci for bootstrap
    """
    high_level = 1 - ci / 2
    low_level = ci / 2
    plot_data_ = (_boot_data
                  .groupby(['group', 'date'])
                  .agg([get_q(low_level), "mean", get_q(high_level), ])
                  )

    for group, g in plot_data_.groupby(level=0):
        plot_g = g.droplevel(level=0)
        plot_g = plot_g.droplevel(level=0, axis=1)
        df = plot_g['mean']
        ax = df.plot(label=f'{group}')
        ax.fill_between(plot_g.index,
                        plot_g[f'q_{low_level}'],
                        plot_g[f'q_{high_level}'], alpha=.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()


class GroupGenerator:
    """ we can generate randomly or predefine from mdp  """
    def __init__(self,
                 start_test=pd.Timestamp(2019, 9, 1),
                 end_test=pd.Timestamp(2020, 6, 15),
                 available_companies=('000000', 'V92599', 'V69852'),
                 narrative=('MODEL_EXP_3',)
                 ):
        self.available_companies = available_companies
        self.end_test = end_test
        self.start_test = start_test
        samples = (get_samples()
                   .assign(start_date=lambda x: pd.to_datetime(x.start_date))
                   .assign(end_date=lambda x: pd.to_datetime(x.end_date))
                   )

        samples = samples[samples['start_date'] <= end_test]
        samples = samples[samples['end_date'] >= start_test]
        if narrative:
            samples = samples[samples['narrative'].isin(narrative)]

        self.samples: pd.DataFrame = samples
        data = all_cnums()[['bundlecode', 'cnum', 'big_city', 'company']]
        if self.available_companies:
            data = data[data['company'].isin(self.available_companies)]
        data = data[~data['company'].isin(COMPANY_BAN)]
        data = data[~data['cnum'].isin(self.samples['CNUM'].unique())]
        data['narrative'] = 'generated'
        self.all_cnums = data
        self.history_data = None
        self.stats = None
        self.data = None

    @staticmethod
    def _generate_test_groups(data,
                              groups=(30_000, 30_000)):
        """ create random group """
        index = np.hstack([np.ones(g) * i for i, g in enumerate(groups)])
        group = (pd.DataFrame(index, columns=['status']))
        data = data.sample(sum(groups))
        data = (pd.concat([data.reset_index(drop=True), group], axis=1)
                .rename(columns={'cnum': "CNUM"}))
        return data

    def generate_test_groups(self,
                             groups=(30_000, 30_000)):
        """ create random group """
        data = self.all_cnums

        return self._generate_test_groups(data, groups)

    def generate_strat_test_groups(self,
                                   groups=(30_000, 30_000)):
        """ create random group stratified by bundle"""
        strats = [('PNEW', 'PREM'), ('GOLD', 'PDIR'),
                  ('STAN', 'BASE', 'NONE')]
        data = self.all_cnums
        # bundle_counts = data.bundlecode.value_counts()
        data_list = []

        for strata in strats:
            # bundle_counts[bundle_counts.index.isin(strata)].sum()
            strata_dt = data[data.bundlecode.isin(strata)]
            bundle_ratio = strata_dt.shape[0] / data.shape[0]
            strata_groups = tuple(int(x * bundle_ratio) for x in groups)
            strata_dt = self._generate_test_groups(strata_dt, strata_groups)
            data_list.append(strata_dt)

        return pd.concat(data_list, axis=0)

    def get_historical_groups(self):
        """ all samples + features from ind. customers """
        all_cnums_not_payroll_raw = all_cnums()[['cnum', 'big_city', 'company']]
        data_df = self.samples

        deals_exp = data_df.merge(all_cnums_not_payroll_raw,
                                  how='left',
                                  left_on='CNUM',
                                  right_on='cnum', )
        return deals_exp

    def set_data(self,
                 start='2019-05-09',
                 end='2019-06-09',
                 ):
        """
        calculate metric for each month from start to end
        """
        deals_exp = get_deals(start, end)
        deals_exp['month'] = deals_exp['deal_date'].dt.to_period('M')
        deals_exp = deals_exp.groupby(["buy_cnum", 'month']).sum().reset_index()
        data = (self.all_cnums[['cnum']]
                .merge(deals_exp, how='left',
                       left_on='cnum',
                       right_on='buy_cnum')
                .pivot_table(index=['cnum'],
                             values=['pnl_rub'],
                             columns='month')
                )

        data.columns = data.columns.droplevel().map(str)
        data.columns.name = None
        self.history_data = data

    def gen_new(self,
                start='2019-05-09',
                end='2019-06-09',
                groups=(90_000,
                        90_000,
                        90_000,
                        90_000)):
        """
        generator for 150 random groups
        """

        for _ in range(150):
            _data = self.generate_strat_test_groups(groups=groups).set_index(
                'CNUM')

            yield _data.join(self.history_data, how='left')

    def gen_with_cross_validation(self,
                                  start='2019-05-09',
                                  end='2019-06-09',
                                  groups=(90_000,
                                          90_000,
                                          90_000,
                                          90_000)):
        """find group with successful  a/a test """

        self.set_data(start, end)

        self.data = [random_g.reset_index().rename(columns={'index': "cnum"})
                     for random_g in self.gen_new(start=start,
                                                  end=end,
                                                  groups=groups)]
        stats = [percents(d.groupby('status').sum()).ewm(alpha=0.3).mean()[-1]
                 for d in self.data]

        best = self.data[np.argmin(stats)]

        # remove generated group from general data
        self.all_cnums = self.all_cnums[~self.all_cnums.cnum.isin(best['CNUM'])]
        return best[['CNUM', 'bundlecode', 'big_city',
                     'company', 'narrative', 'status']]


class ABTest:
    """
    plot metrics
    """

    amounts = [0, 7e4, 3e5, 1e6, 6.5e6, float('inf')]  # buckets

    bundles = ['PNEW', 'PREM', 'GOLD', 'PDIR', ]

    def __init__(self,
                 data_df,
                 start_deal_date='2020-05-18',
                 end_deal_date='2020-05-28',
                 special_rates=None
                 ):

        self.end_deal_date = end_deal_date
        self.start_deal_date = start_deal_date
        self.only_deals = self._get_deals()
        self._deals_exp = pd.DataFrame()
        self.special_rates = special_rates
        self.mask = None
        self.set_groups(data_df)

    def _get_deals(self):
        return get_deals(self.start_deal_date, self.end_deal_date)

    @classmethod
    def quantilize_minimal(cls, df):
        """ labels for buckets  """
        return (pd.cut(df['rub_amount'],
                       cls.amounts,
                       labels=False,
                       retbins=False)
                .fillna(-1).astype(int))

    @classmethod
    def reduce_bundles(cls, df):
        """ small bundles is not important """
        return (df['bundlecode']
                .where(df['bundlecode'].isin(cls.bundles), 'NONE')
                )

    @property
    def deals_exp(self) -> pd.DataFrame:
        """for cache _deals_exp"""
        return self._deals_exp[self.mask]

    def set_groups(self, data):
        """ merge with deals and create buckets and etc."""
        deals_exp = (data
                     .merge(self.only_deals,
                            how='left',
                            left_on='CNUM',
                            right_on='buy_cnum',
                            )
                     )
        deals_exp = (deals_exp
                     .assign(bundlecode=self.reduce_bundles)
                     .assign(amount_bucket=self.quantilize_minimal)
                     .assign(active=lambda df: (df['pnl_rub']
                                                .notnull()))
                     )
        self._deals_exp = deals_exp
        self.set_mask(narrative=tuple())

    def set_mask(self,
                 amount: int = 5,
                 bundlecode: Tuple = tuple(),
                 narrative: Tuple = tuple(),
                 status: Tuple = ('churn_85%', 'churn_control')
                 ):
        """
        :param status: not in ('churn_85%', 'churn_control', dop_control ....)
        :param narrative: in [  'movehist_exp',
                                'stealth_exp',
                                'EXP_PAYROLL_1',
                                'exp_3'
                                'MODEL_EXP_3',
        :param amount: < amounts
                                0:7e4,
                                1:3e5,
                                2:1e6,
                                3:6.5e6,
                                4:float('inf')
        # buckets
        :param bundlecode: not in bundles ['PNEW', 'PREM','GOLD','PDIR']
        """

        mask = self._deals_exp['amount_bucket'] <= amount

        # personal discount

        def filter_special(df):
            specials = self.special_rates
            if df['buy_cnum'] not in specials['cnum'].values:
                return True
            specials = specials[specials['cnum'] == df['buy_cnum']]
            specials = specials[specials['start_date'] <= df['deal_date']]
            specials = specials[specials['end_date'] > df['deal_date']]
            return specials.empty

        if self.special_rates is not None:
            mask &= self._deals_exp.apply(filter_special, axis=1)

        if status:
            mask &= (~self._deals_exp['status'].isin(status))
        if bundlecode:
            mask &= (~self._deals_exp['bundlecode'].isin(bundlecode))
        if narrative:
            mask &= (self._deals_exp['narrative'].isin(narrative))

        self.mask = mask

    @property
    def deals_gp_buckets(self):
        """
        Product metrics
        """

        deals_gp_buckets = pd.get_dummies(self.deals_exp,
                                          columns=['amount_bucket'])

        name_for_buckets = {f'amount_bucket_{i}': f'{naming(v)}'
                            for i, v in enumerate(self.amounts[1:])}

        dict_agg = {
            **{
                'pnl_rub':    ['sum'],
                'rub_amount': ['sum', 'mean', get_q(0.3), 'median',
                               get_q(0.7), get_q(0.9)],
                'buy_cnum':   [uniq, 'count'],
                'CNUM':       [uniq],
            },
            # all Buckets after filtering
            **{col: 'sum' for col in deals_gp_buckets.columns
               if col.startswith('amount_bucket_')
               # all Nones with out deals
               and col != 'amount_bucket_-1'
               }
        }

        def margin_percent(df):
            """pnl/amount"""
            return df['pnl_rub', 'sum'] / df['rub_amount', 'sum'] * 100

        def pnl_per_user(df):
            """pnl/all active"""
            return df['pnl_rub', 'sum'] / df['CNUM', 'uniq']

        def ctr_percent(df):
            """all active/ group size"""
            return df['buy_cnum', 'uniq'] / df['CNUM', 'uniq'] * 100

        return (deals_gp_buckets
                .groupby(['status'])
                .agg(dict_agg)
                .rename(columns=name_for_buckets)
                .assign(margin_perc=margin_percent)
                .assign(pnl_per_user=pnl_per_user)
                .assign(ctr_perc=ctr_percent))

    def print_product_metrics(self):
        """ table with pnl """
        pd.set_option('float_format', '{:,.0f}'.format)
        print(self.deals_gp_buckets.iloc[:, :7])

        pd.set_option('float_format', '{:,.2f}'.format)
        print(self.deals_gp_buckets.iloc[:, 7:].astype(float))


def percents(df):
    """inline func calculate percents"""
    return (df.max() - df.min()) / df.mean()


def prep(df):
    """inline func look 2 type groups separately"""
    b = df.loc[4:, 'pnl_rub']
    a = df.loc[:3, 'pnl_rub']
    return percents(a), percents(b)
