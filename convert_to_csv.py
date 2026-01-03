from pickle import TRUE
import joblib
import pandas as pd
import os

obj = pd.read_pickle('/Users/kahrens/MyProjects/Dube_replication_condensed/doc2vec_output/ipeirotis_doc2vec.pkl')
obj.to_csv('/Users/kahrens/MyProjects/Dube_replication_condensed/doc2vec_output/ipeirotis_doc2vec.csv')
