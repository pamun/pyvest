from pyvest.factor_model.factor_model import FactorModel


class FF3F(FactorModel):
    def __init__(self, factors, returns, r_f_column="RF",
                 mkt_rf_column="Mkt-RF", smb_column="SMB", hml_column="HML",
                 name="FF3F"):

        r_f = factors[r_f_column]
        ff3f_factors = factors[[mkt_rf_column, smb_column, hml_column]]

        super().__init__(r_f, ff3f_factors, returns, name)
