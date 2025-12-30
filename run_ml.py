######
### Run from the command line like:
### python run_ml.py <dataset> <mode>
### For example:
### python run_ml.py textlab_10 textab
######

# Python imports
import argparse
import logging
import os
import time

# 3rd party imports
import pandas as pd
import numpy as np
import joblib

# Scikit imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

# Internal imports
from global_settings import output_paths

ml_input_path = output_paths["ml_input"]
ml_output_path = output_paths["ml_output"]
    
# Quick decorator to print runtime of functions
# www.laurivan.com/braindump-use-a-decorator-to-time-your-python-function/
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        #print('%r (%r, %r) %2.2f sec' % (method.__name__, args, kw, te-ts))
        print(method.__name__ + ": " + str(te-ts) + " seconds")
        return result
    return timed
  
# Function for quickly running a classifier and saving results
@timeit
def fitAndPredict(X_train, y_train, X_test, y_test, classifier, dataset_name, y_name):
    print("Running " + dataset_name + "_" + y_name)
    # Use this function to compare R^2 among a bunch of different regression
    # methods (just pass in different classifiers)
    fitted_reg = classifier.fit(X_train, y_train)
    print(dataset_name + "_" + y_name + " fit complete")

    # Save the fitted model to disk
    fitted_filename = os.path.join(ml_output_path, "fitted_" + dataset_name + "_" + y_name + ".pkl")
    joblib.dump(fitted_reg, fitted_filename)
    print("Fitted model saved")
    # Get and save predicted values
    y_predicted = fitted_reg.predict(X_test)
    joblib.dump(y_predicted, os.path.join(ml_output_path, "predictions_" + dataset_name + "_" + y_name + ".pkl"))
    # Compute and return prediction score
    pred_score = fitted_reg.score(X_test, y_test)
    joblib.dump(pred_score, os.path.join(ml_output_path, "pred_score_" + dataset_name + "_" + y_name + ".pkl"))
    return pred_score
  
def runML(dataset, mode):
    #num_cpus = multiprocessing.cpu_count()
    rewname = mode + "_rew"
    durname = mode + "_dur"
    text_trees = 40
    numeric_trees = 600
    
    #
    # For running on AWS [REMEMBER: run in a tmux session so that it keeps running
    # if you get disconnected! Also use htop to check that all processors are
    # being utilized]:
    # 
    # The input files needed for *text* ML are:
    # (1a) train_txtfeats_<dataset>_<mode>.pkl, train_rew_<dataset>_<mode>.pkl, and train_dur_<dataset>_<mode>.pkl
    # (1b) test_txtfeats_<dataset>_<mode>.pkl, test_rew_<dataset>_<mode>.pkl, and test_dur_<dataset>_<mode>.pkl
    # 
    # Upon completion of *text* ML, the program will output:
    # (1c) fitted_rfr_rewtext.pkl, predictions_rfr_rewtext.pkl, and pred_score_rfr_rewtext.pkl
    # (1d) fitted_rfr_durtext.pkl, predictions_rfr_durtext.pkl, and pred_score_rfr_durtext.pkl
    #
    # The input files needed for *numeric* ML are:
    # (2a) train_feats_<dataset>_<mode>.pkl, train_rew_<dataset>_<mode>.pkl, and train_dur_<dataset>_<mode>.pkl
    # (2b) test_feats_<dataset>_<mode>.pkl, test_rew_<dataset>_<mode>.pkl, and test_dur_<dataset>_<mode>.pkl.
    #
    # Upon completion of *numeric* ML, the program will output:
    # (2c) fitted_<dataset>_<mode>_rew.pkl, predictions_<dataset>_<mode>_rew.pkl, and pred_score_<dataset>_<mode>_rew.pkl
    # (2d) fitted_<dataset>_<mode>_dur.pkl, predictions_<dataset>_<mode>_dur.pkl, and pred_score_<dataset>_<mode>_dur.pkl
    #
    
    ### Load training and test data
    if mode.startswith("gram"):
        ## X_train text features
        X_train = joblib.load(os.path.join(ml_input_path, 
            "train_txtfeats_" + dataset + "_" + str(mode) + ".pkl"))
    else:
        ## X_train numeric features
        X_train_df = pd.read_pickle(os.path.join(ml_input_path, 
            "train_feats_" + dataset + "_" + str(mode) + ".pkl"))
        X_train = X_train_df.values
    ## y_train_rew
    y_train_rew_df = pd.read_pickle(os.path.join(ml_input_path, 
        "train_rew_" + dataset + "_" + mode + ".pkl"))
    y_train_rew = y_train_rew_df.values
    ## y_train_dur
    y_train_dur_df = pd.read_pickle(os.path.join(ml_input_path, 
        "train_dur_" + dataset + "_" + mode + ".pkl"))
    y_train_dur = y_train_dur_df.values
    print("Training data loaded")
    
    if mode.startswith("gram"):
        ## X_test text features
        X_test = joblib.load(os.path.join(ml_input_path, 
            "test_txtfeats_" + dataset + "_" + mode + ".pkl"))
    else:
        ## X_test numeric features
        X_test_df = pd.read_pickle(os.path.join(ml_input_path, 
            "test_feats_" + dataset + "_" + mode + ".pkl"))
        X_test = X_test_df.values
    ## y_test_rew
    y_test_rew_df = pd.read_pickle(os.path.join(ml_input_path,
        "test_rew_" + dataset + "_" + mode + ".pkl"))
    y_test_rew = y_test_rew_df.values
    ## y_test_dur
    y_test_dur_df = pd.read_pickle(os.path.join(ml_input_path, 
        "test_dur_" + dataset + "_" + mode + ".pkl"))
    y_test_dur = y_test_dur_df.values
    print("Test data loaded")
    
    ### Now use fitAndPredict() to compare different methods
    # knn is good but slow. gbr does worse than rfr but is quicker.
    # rfr is best by far but also slow. but is parallelized!
    if mode.startswith("gram"):
        ## Text feature ML
        # y = reward
        print("----- [Running *reward* ML with text features] -----")
        reward_rfr = RandomForestRegressor(n_jobs=-1,oob_score=False,
            n_estimators=text_trees,verbose=2)
        reward_r2 = fitAndPredict(X_train,y_train_rew,X_test,y_test_rew,
            reward_rfr,dataset,rewname)
        print(reward_r2)
        # y = duration
        print("----- [Running *duration* ML with text features] -----")
        duration_rfr = RandomForestRegressor(n_jobs=-1,oob_score=False,
            n_estimators=text_trees,verbose=2)
        duration_r2 = fitAndPredict(X_train,y_train_dur,X_test,y_test_dur,
            duration_rfr,dataset,durname)
        print(duration_r2)

    else:
        ## Numeric feature ML
        print("----- [Running *reward* ML with numeric features] -----")
        reward_rfr = RandomForestRegressor(n_jobs=-1,oob_score=True,
            n_estimators=numeric_trees,verbose=2)
        reward_r2 = fitAndPredict(X_train,y_train_rew,X_test,y_test_rew,
            reward_rfr,dataset,rewname)
        print(reward_r2) 
        print("----- [Running *duration* ML with numeric features] -----")
        duration_rfr = RandomForestRegressor(n_jobs=-1,oob_score=True,
            n_estimators=numeric_trees,verbose=2)
        duration_r2 = fitAndPredict(X_train,y_train_dur,X_test,y_test_dur,
            duration_rfr,dataset,durname)
        print(duration_r2)

    # For latency/throughput comparisons among the methods see:
    # http://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html

def main():
    # Enable logging to stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("mode")
    args = parser.parse_args()
    runML(args.dataset, args.mode)

#if __name__ == "__main__":
#    main()
