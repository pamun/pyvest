from pyvest.factor_model.factor_model import FactorModel


class CAPM(FactorModel):
    def __init__(self, factors, returns, r_f_column="RF",
                 mkt_rf_column="Mkt-RF", name="CAPM"):

        r_f = factors[r_f_column]
        capm_factors = factors[mkt_rf_column]

        super().__init__(r_f, capm_factors, returns, name)

    def plot(self, compare_with=None, labels=None, colors=None,
             error_bars_colors=None, legend='upper left', min_return=0,
             max_return=1.5, sml=True, beta_min=0, beta_max=2):
        super().plot(compare_with=compare_with, labels=labels, colors=colors,
                     error_bars_colors=error_bars_colors, legend=legend,
                     min_return=min_return, max_return=max_return, sml=sml,
                     beta_min=beta_min, beta_max=beta_max)
