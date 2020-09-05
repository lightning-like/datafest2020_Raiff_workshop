"""
boot
"""
import pandas as pd
from matplotlib import pyplot as plt

from fxpi_ml.utl.log import configure_logger

LOGGER = configure_logger(__name__)


class BootFunction:
    """
    base class for boot calculation
    you need rewrite bootstrap method for other realization
    """

    def __str__(self):
        return f"Standard bootstrap on {self.metric}"

    __repr__ = __str__

    def __init__(self,
                 metric='pnl_rub',
                 stratification=(),
                 p_val=0.2,
                 *args,
                 **kwargs):
        """
        :param metric: column from ABTest data
        :param stratification: can be any columns from data
        """

        self.p_val = p_val
        self.stratification = list(stratification)
        self.metric = metric
        self._population = None
        self.status_size = None
        self.size_ratio = None

    def prepare_data(self, data):
        """
        in case you need change all data in begging
        """
        groups = ['deal_date', 'status', 'CNUM']
        deals = data
        data = (data[groups + [self.metric]]
                .dropna(subset=['deal_date'])
                .set_index(groups)
                .fillna(0)
                .sort_index())

        min_date = min(data.index.get_level_values('deal_date'))
        max_date = max(data.index.get_level_values('deal_date'))
        multi_ind = pd.date_range(min_date, max_date, freq='D')
        for s, data_status in data.groupby('status'):
            self.set_sampled_cnums(deals, s)
            self._population = self._population[self._population.index.isin(
                data.index.get_level_values(2))]
            list_data = list(self._get_strat_data(data_status, multi_ind))
            pd_data = (pd.concat(list_data, ignore_index=True)
                       .iloc[::-1]
                       .reset_index(drop=True)
                       .set_index(pd.Index(multi_ind[1:])))
            yield s, pd_data

    def _get_strat_data(self, df, multi_ind):
        """
        create generator for each stratification group
        index dates
        columns must have "CNUM" on level 0
        drop level = 0 and return only stratification in index
        """

        for idate in reversed(multi_ind[1:]):
            days = (idate - multi_ind[0]).days

            yield self(df.loc[:idate], days)

    def __call__(self, deals_exp, days):

        data = (deals_exp
                .groupby(by=['CNUM'])
                [self.metric]
                .sum()
                )
        return self.bootstrap(data / days)

    def bootstrap(self, data):
        """
        standard bootstrap
        """

        boot_data = (self._population
                         .join(data, how='left')
                         .groupby('group')
                         .sum()
                         .iloc[:, 0])
        return pd.DataFrame([[boot_data.mean(),
                              boot_data.quantile(
                                  q=self.p_val),
                              boot_data.quantile(
                                  q=1 - self.p_val), ]],
                            columns=('value', 'low', 'up')) / self.size_ratio

    def set_sampled_cnums(self, data, status):
        """
        sample population for each group(not change in time)
        """

        data = data[data.status == status]

        status_cnums = data['CNUM'].drop_duplicates()
        self.size_ratio = status_cnums.shape[0] / 15000
        n = 1000

        groups = list(range(n))
        groups = (pd.Series(groups * len(status_cnums), name='group')
                  .sort_values()
                  .reset_index(drop=True)
                  )
        # separate sampling groups * n
        boot_data = (status_cnums
                     .reset_index(drop=True)
                     .sample(frac=n, replace=True)
                     .reset_index(drop=True)
                     )
        # create n samples for each group
        boot_data = (pd.concat([boot_data, groups], axis=1)
                     .set_index('CNUM'))

        self._population = boot_data


def get_cuped(
        metrics: pd.Series,
        covariat: pd.Series,
        ):
    """
    :param covariat: Historical data
    :param metrics: current data
    """
    new_metric = (covariat - covariat.mean())
    new_metric *= covariat.cov(metrics) / covariat.var()
    new_metric *= -1
    new_metric += metrics
    return new_metric


class Stratification(BootFunction):
    """
    stratification by index
    we bootstrap separately all groups
    """

    def set_sampled_cnums(self, data, status):
        """
        sample population for each group(fixed in time)
        """
        if not self.stratification:
            super().set_sampled_cnums(data, status)
            return
        data = data[data.status == status]
        status_cnums = data[['CNUM'] + self.stratification].drop_duplicates()
        self.size_ratio = status_cnums.shape[0] / 15000
        stratification_data = []
        for _, group_ in status_cnums.groupby(self.stratification):
            n = 1000
            # marker for futures summing
            groups = list(range(n))
            groups = (pd.Series(groups * len(group_), name='group')
                      .sort_values()
                      .reset_index(drop=True)
                      )
            # separate sampling groups * n
            boot_data = (group_['CNUM']
                         .reset_index(drop=True)
                         .sample(frac=n, replace=True)
                         .reset_index(drop=True)
                         )
            # create n samples for each group
            boot_data = (pd.concat([boot_data, groups], axis=1)
                         .set_index('CNUM'))

            stratification_data.append(boot_data)
        stratification_data = pd.concat(stratification_data, axis=0)
        self._population = stratification_data


class Cuped(Stratification):
    """
    Create function for bootstrap with Cuped data
    """

    def __init__(self,
                 metric='pnl_rub',
                 stratification=tuple(),
                 cuped_data=None,
                 p_val=0.05,
                 *args, **kwargs):
        super().__init__(metric, stratification, p_val=p_val, *args, **kwargs)
        if cuped_data is None:
            LOGGER.error("please use base standard bootstrap if u have not "
                         "historical data")

        self._cuped_data = cuped_data.dropna(subset=['deal_date'])
        min_date = min(self._cuped_data['deal_date'])
        max_date = max(self._cuped_data['deal_date'])
        self._cuped_data = (self._cuped_data
                            .groupby(['CNUM', 'status'])
                            [self.metric].sum()
                            / (max_date - min_date).days)
        self.cuped_data = None

    def bootstrap(self, data: pd.Series):
        """

        :param data:
        :return:
        """
        data = (pd.concat([data, self.cuped_data],
                          axis=1,
                          ignore_index=True
                          )
                # ? do we need look on non active users ?
                .dropna(subset=[0])
                .fillna(0)
                )
        boot_data = (self._population
                     .join(data,
                           how='left')
                     .groupby('group')
                     .sum())
        boot_data = get_cuped(boot_data.iloc[:, 0],
                              boot_data.iloc[:, 1])
        return pd.DataFrame([[boot_data.mean(),
                              boot_data.quantile(
                                  q=self.p_val),
                              boot_data.quantile(
                                  q=1 - self.p_val), ]],
                            columns=('value', 'low', 'up')) / self.size_ratio

    def set_sampled_cnums(self, data, status):
        """
        sample population for each group(fixed in time)
        """

        super().set_sampled_cnums(data, status=status)
        self.cuped_data = (
            self._cuped_data[self._cuped_data.index.get_level_values(1) ==
                             status].reset_index(level=1, drop=True))


class BootAB:
    """
    estimate variance on ab testing
    """

    def __init__(self,
                 deals_exp,
                 boot_func=BootFunction(),
                 ):
        """

        :param deals_exp: data from ABTest
        :param boot_func: can be Cuped(), stratification_boot, general_boot
        """

        self.deals_exp = deals_exp
        self.boot_func = boot_func
        self.boot_metric = {}

    def format_print(self, fmt='jira'):
        """
        table of last ci
        """
        data = pd.concat([i.tail(1)
                          for i in self.boot_metric.values()],
                         ignore_index=True
                         )
        data.index = list(self.boot_metric.keys())
        data['std'] = 100 * (data['up'] - data['low']) / data['value']
        print(data.applymap(lambda x: f'{float(x):,.0f}')
              .to_markdown(tablefmt=fmt))

    def plot(self):
        """
        Plot ci
        """
        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 10})
        for label, df in self.boot_metric.items():
            ax = df.value.plot(label=f'{label}')
            ax.fill_between(df.index,
                            df['low'],
                            df['up'], alpha=.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   borderaxespad=0.)
        plt.title(f'{self.boot_func}')
        plt.show()

    def create_boot_data(self):
        """
        run boot_func on date
        """
        self.boot_metric.clear()
        for s, pd_data in self.boot_func.prepare_data(self.deals_exp):
            self.boot_metric[f'{s}'] = pd_data
