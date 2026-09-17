"""Checks for errors that would invalidate temporal transfer results."""
import unittest
import numpy as np
import pandas as pd
from src.transfer_candidates import panel, CATS, metrics


class TransferDataTests(unittest.TestCase):
    def sample(self, periods=(8084,8085,8086)):
        return pd.DataFrame([{'region':'A','period':p,'category':c,'amount':100*(i+1),'count':10*(i+1)}
                             for i,p in enumerate(periods) for c in CATS])

    def test_adjacent_growth_and_unknown_future(self):
        p=panel(self.sample())
        self.assertTrue(np.isnan(p.iloc[0].amount_growth))
        self.assertEqual(p.iloc[1].amount_growth,1)
        self.assertEqual(p.iloc[1].future_growth,.5)
        self.assertTrue(np.isnan(p.iloc[-1].target))

    def test_missing_quarter_is_not_previous_or_next_quarter(self):
        p=panel(self.sample((8084,8086)))
        self.assertTrue(p.amount_growth.isna().all())
        self.assertTrue(p.future_growth.isna().all())

    def test_missing_category_is_not_zero_sales(self):
        d=self.sample()
        d=d[~((d.period==8085)&(d.category==CATS[0]))]
        p=panel(d)
        self.assertTrue(np.isnan(p.iloc[1].amount))
        self.assertTrue(np.isnan(p.iloc[0].future_growth))
        self.assertTrue(np.isnan(p.iloc[2].amount_growth))

    def test_top_k_is_selected_within_each_quarter(self):
        d=pd.DataFrame({'period':[1]*5+[2]*5,'target':[1,0,0,0,0,1,0,0,0,0]})
        result=metrics(d,[10,9,8,7,6,1,0,0,0,0],'example','test')
        self.assertEqual(result['top20_precision_by_quarter'],1)


if __name__=='__main__':
    unittest.main()
