_base_ = ['./mgnn_depression_avec2014_res50.py']


test_evaluator = dict(type='RegMetric', dataset="AVEC2014")