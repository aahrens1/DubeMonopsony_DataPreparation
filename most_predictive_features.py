# Just a quick script that generates .csvs with the most predictive features
# for reward and duration
import argparse
import os

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from global_settings import output_paths

ml_input_path = output_paths["ml_input"]
ml_output_path = output_paths["ml_output"]
predictive_path = output_paths["predictive_feats"]

def computeMostPredictive(dataset, mode):
    for depvar in ["dur","rew"]:
        print("Generating most predictive features for " + depvar)
        fitted_filepath = os.path.join(ml_output_path, "fitted_" + dataset + "_" + mode + "_" + depvar + ".pkl")
        rfr = joblib.load(fitted_filepath)
        feature_names_filepath = os.path.join(ml_input_path, "feat_names_" + dataset + "_" + mode + ".pkl")
        feature_names = joblib.load(feature_names_filepath)
        num_features = len(feature_names)
        print("Number of features: " + str(num_features))

        feature_dict = {feat_index: feat_name for feat_index, feat_name in list(enumerate(feature_names))}
        # The importances are a big array holding the gini score for each feature.
        # The thing to note here is that we hstacked the desc and title features
        # together, so when interpreting this array we need to view all indices
        # above len(desc_features) to be indexing title_features
        importances = rfr.feature_importances_
        sorted_imp = importances.argsort()
        sorted_imp = list(reversed(sorted_imp))
        
        ## For numeric features
        #imp_arr = [(ind, feature_dict[ind], importances[ind]) for ind in sorted_imp]
        #print(imp_arr)
        
        ## For text features
        sorted_feature_names = [feature_names[ind] for ind in sorted_imp]
        sorted_importances = [importances[ind] for ind in sorted_imp]
        importance_df = pd.DataFrame({'feature':sorted_feature_names,'gini':sorted_importances})
        # array of (feature_index, feature_name, feature_importance) tuples
        #imp_arr = [feature_names[ind] + ": " + str(importances[ind]) for ind in sorted_imp]
        #print(importance_df.head(300).to_csv())
        csv_filepath = os.path.join(predictive_path,"predictive_" + dataset + "_" + mode + "_" + depvar + ".csv")
        importance_df.head(300).to_csv(csv_filepath)
        print("Features outputted to: " + csv_filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("mode")
    args = parser.parse_args()
    computeMostPredictive(args.dataset, args.mode)

if __name__ == "__main__":
    main()