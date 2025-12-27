######
### This is the .py version of what used to be in the MTurk_ML.ipynb notebook
### 
### Run from the command line like:
### python prepare_ml.py <dataset> <mode>
### For example:
### python prepare_ml.py textlab_10 gramab
### You can also add a --clean flag to re-run the data cleaning:
### python prepare_ml.py textlab_10 gramab --clean
###
### The datasets are: textlab_10, textlab_30, ipeirotis
### The modes are: gramab, numab, gramba, numba
######
import argparse
import os
import re

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

from global_settings import output_paths

cleaned_path = output_paths["cleaned"]
doc2vec_path = output_paths["doc2vec"]
lda_path = output_paths["lda"]
predictive_path = output_paths["predictive_feats"]
ml_input_path = output_paths["ml_input"]

def loadCleanedData(dataset):
    if dataset == "textlab_30":
        hit_df = pd.read_pickle(os.path.join(cleaned_path, "textlab_30_cleaned.pkl"))
        hit_df["title"] = hit_df["title"].astype(str)
    elif dataset == "textlab_10":
        hit_df = pd.read_pickle(os.path.join(cleaned_path, "textlab_10_cleaned.pkl"))
        # This is hella annoying
        hit_df["description"] = hit_df["description"].astype(str)
    elif dataset == "ipeirotis":
        hit_df = pd.read_pickle(os.path.join(cleaned_path, "ipeirotis_cleaned.pkl"))
        hit_df["description"] = hit_df["description"].astype(str)
        hit_df["title"] = hit_df["title"].astype(str)
    
    ## Couldn't find a better (less awkward) place to put this
    # Compute mean-differenced rewards (the requester means will be used as features)
    hit_df["req_mean_reward"] = hit_df.groupby("requester_id")["reward"].transform("mean")
    hit_df["log_req_mean_reward"] = hit_df["req_mean_reward"].apply(np.log)
    hit_df["meandiff_lreward"] = hit_df["log_reward"] - hit_df["log_req_mean_reward"]

    # Compute mean-differenced durations (the requester means will again be used as features)
    hit_df["req_mean_dur"] = hit_df.groupby("requester_id")["duration"].transform("mean")
    hit_df["log_req_mean_dur"] = hit_df["req_mean_dur"].apply(np.log)
    hit_df["meandiff_ldur"] = hit_df["log_duration"] - hit_df["log_req_mean_dur"]
    return hit_df

def partitionData(hit_df):
    ## Generates the indices for the train/trainval/test/testval split
    # First generate a unique numerical index (0,1,2,...)
    hit_df["group_num"] = range(0,len(hit_df))

    # Split the data into A and B halves. See:
    #http://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
    test_prop = 0.5
    all_indices = hit_df["group_num"].tolist()
    A_ind, B_ind = train_test_split(all_indices, test_size=test_prop, random_state=48)
    print("# A obs: " + str(len(A_ind)))
    print("# B obs: " + str(len(B_ind)))

    # And now sub-split each of the two sets again into training+validation
    # (Validation here is the "test" data for learning the textual features)
    validation_prop = 0.2
    A_train_ind, A_val_ind = train_test_split(A_ind, test_size=validation_prop, random_state=49)
    B_train_ind, B_val_ind = train_test_split(B_ind, test_size=validation_prop, random_state=50)

    print("# A_train obs: " + str(len(A_train_ind)))
    print("# A_val obs: " + str(len(A_val_ind)))
    print("# B_train obs: " + str(len(B_train_ind)))
    print("# B_val obs: " + str(len(B_val_ind)))
    return {'A':A_ind,'B':B_ind,'A_train':A_train_ind,'A_val':A_val_ind,'B_train':B_train_ind,'B_val':B_val_ind}

##################################
### NUMERIC FEATURE EXTRACTION ###
##################################

def generateNumericFeatures(hit_df, dataset, mode, partition):
    # Vars specific to feature extraction
    # Indicative words map. Maps variable name to the regex for that variable
    # Some explanations: "only" typically indicates that it's restricted to some group
    # (e.g., US only, 18-29 only)
    indicative_map = {"easy":"easy","transcribe":"transcr",
                      "writing":"writ","audio":"audio","image":"image|picture","video":"video","bonus":"bonus",
                     "copy":"copy","search":"search","identify":"ident","text":"text",
                     "date":"date","fun":"fun","simple":"simpl","summarize":"summar","only":"only",
                     "improve":"improve","five":"five|5","questionmark":"\?","exclamation":"!"}

    # NEW: we load the features from the .csv files
    desc_top = {}
    title_top = {}
    kw_top = {}
    feature_reg = r'\"(.+)\"'
    def updateFeatureDict(feature_list, feature_source):
        for feature_num, cur_feature in enumerate(feature_list):
            if feature_num >= 100:
                # Only save the top 100 features
                break
            feature_str = re.findall(feature_reg, cur_feature)[0]
            feature_suffix = feature_source + str(feature_num)
            if cur_feature.endswith("description"):
                desc_top[feature_suffix] = feature_str
            elif cur_feature.endswith("title"):
                title_top[feature_suffix] = feature_str
            else:
                kw_top[feature_suffix] = feature_str
                
    # mode_order is just "ab" or "ba"
    mode_order = mode[-2:]
    rew_feat_df = pd.read_csv(os.path.join(predictive_path, 
            "./predictive_" + dataset + "_gram" + mode_order + "_rew.csv"))
    rew_feat_list = rew_feat_df["feature"].tolist()
    updateFeatureDict(rew_feat_list, "r")

    dur_feat_df = pd.read_csv(os.path.join(predictive_path, 
            "./predictive_" + dataset + "_gram" + mode_order + "_dur.csv"))
    dur_feat_list = dur_feat_df["feature"].tolist()
    updateFeatureDict(dur_feat_list, "d")

    print("Number of description features: " + str(len(desc_top)))
    print("Number of title features: " + str(len(title_top)))
    print("Number of keyword features: " + str(len(kw_top)))

    gadiraju_map = {"find":"find", "check":"check", "match":"match", "choose":"choose", "categorize":"categor",
                    "suggest":"suggest","translate":"translat","survey":"survey","click":"click","link":"link",
                   "read":"read"}
    gadiraju_categories = {'IF':['find'], 'VV':["check","match"], 'IA':['choose','categorize'],
                            'CC':['suggest','translate'], 'S':['survey'], 'CA':['click','link','read']}
    

    # Now combine them
    indicative_map.update(gadiraju_map)
    # Qualification regexes
    if dataset == "textlab_10":
        list_sep = "\|"
    else:
        list_sep = "',"

    ## Now the actual computation
    parseTimes(hit_df, dataset)
    # Make a new df just for the numeric features
    feature_df = hit_df[["duration","log_duration","reward","log_reward",
                         "meandiff_lreward","meandiff_ldur","time_allotted"]].copy()
    computeSimpleFeatures(hit_df, feature_df, dataset, list_sep)
    computeIndicatorFeatures(hit_df, feature_df, dataset, indicative_map, 
        gadiraju_categories, desc_top, title_top, kw_top)
    if dataset == "ipeirotis":
        computeQualificationFeatures(hit_df, feature_df, list_sep)
    mergeLDA(feature_df, dataset)
    mergeDoc2Vec(feature_df, dataset)
    checkNulls(feature_df)
    fixNulls(feature_df)

    # First create the train-test dataframes
    if mode.endswith("ab"):
        train_df = feature_df.iloc[partition['A']]
        test_df = feature_df.iloc[partition['B']]
    else:
        train_df = feature_df.iloc[partition['B']]
        test_df = feature_df.iloc[partition['A']]
    # Now define the non-feature cols that will be dropped when exporting feature file
    drop_cols = ["group_id","reward","log_reward","duration","log_duration","meandiff_lreward",
             "meandiff_ldur"]

    #####################
    ### TRAINING DATA ###
    #####################

    ### First we export the labels (the order is important because we drop the labels
    ### when exporting features in order to save memory - rather than making a new df)

    ## Training data: y1(rewards)
    train_labels_rew = train_df["log_reward"]
    train_rewards_filename = os.path.join(ml_input_path, 
        "./train_rew_" + dataset + "_" + str(mode) + ".pkl")
    train_labels_rew.to_pickle(train_rewards_filename)
    print("Train rewards saved to " + train_rewards_filename)
    ## Training data: y2(duration)
    train_labels_dur = train_df["log_duration"]
    train_durations_filename = os.path.join(ml_input_path,
        "./train_dur_" + dataset + "_" + str(mode) + ".pkl")
    train_labels_dur.to_pickle(train_durations_filename)
    print("Train durations saved to " + train_durations_filename)

    ## Training data: numeric features
    train_features_filename = os.path.join(ml_input_path,
        "./train_feats_" + dataset + "_" + str(mode) + ".pkl")
    # Drop unneeded columns
    for col_name in drop_cols:
        if col_name in train_df:
            train_df.drop([col_name],axis=1,inplace=True)
    joblib.dump(train_df.columns, os.path.join(ml_input_path,
        "./feat_names_" + dataset + "_" + str(mode) + ".pkl"))
    train_df.to_pickle(train_features_filename)
    print("Train features saved to " + train_features_filename)
    print("Training data saved")

    #################
    ### TEST DATA ###
    #################

    ## Test data: y1(rewards)
    test_labels_rew = test_df["log_reward"]
    test_rewards_filename = os.path.join(ml_input_path,
        "./test_rew_" + dataset + "_" + str(mode) + ".pkl")
    test_labels_rew.to_pickle(test_rewards_filename)
    print("Test rewards saved to " + test_rewards_filename)
    ## Test data: y2(duration)
    test_labels_dur = test_df["log_duration"]
    test_durations_filename = os.path.join(ml_input_path,
        "./test_dur_" + dataset + "_" + str(mode) + ".pkl")
    test_labels_dur.to_pickle(test_durations_filename)
    print("Test durations saved to " + test_durations_filename)

    ## Test data: numeric features
    test_features_filename = os.path.join(ml_input_path,
        "./test_feats_" + dataset + "_" + str(mode) + ".pkl")
    # Unneeded columns drop
    for col_name in drop_cols:
        if col_name in test_df:
            test_df.drop([col_name],axis=1,inplace=True)
    test_df.to_pickle(test_features_filename)
    print("Test features saved to " + test_features_filename)
    print("Test data saved")

    print("Training data dimensions: " + str(train_df.shape))
    print("Test data dimensions: " + str(test_df.shape))

def parseTimes(hit_df, dataset):
    ## Convert times from string format (e.g., "30 minutes") to int
    min_reg = r'([0-9]+) (?:minute)|(?:minutes)'
    hr_reg = r'([0-9]+) (?:hour)|(?:hours)'
    day_reg = r'([0-9]+) (?:day)|(?:days)'
    week_reg = r'([0-9]+) (?:week)|(?:weeks)'
    #month_reg = r'([0-9]+) (?:month)|(?:months)'
    #yr_reg = r'([0-9]+) (?:year)|(?:years)'

    def convertToMins(cur_time):
        if type(cur_time) == int:
            # It has already been converted
            return cur_time
        total_mins = 0
        # Minutes
        min_match = re.findall(min_reg, cur_time)
        num_mins = int(min_match[0]) if min_match else 0
        total_mins += num_mins
        # Hours
        hr_match = re.findall(hr_reg, cur_time)
        num_hrs = int(hr_match[0]) if hr_match else 0
        total_mins += 60*num_hrs
        # Days
        day_match = re.findall(day_reg, cur_time)
        num_days = int(day_match[0]) if day_match else 0
        total_mins += 60*24*num_days
        # Weeks
        week_match = re.findall(week_reg, cur_time)
        num_weeks = int(week_match[0]) if week_match else 0
        total_mins += 60*24*7*num_weeks
        return total_mins

    if dataset == "textlab_10" or dataset == "textlab_30":
        # Gotta parse time_allotted and time_left
        hit_df["time_allotted"] = hit_df["time_allotted"].apply(convertToMins)
        hit_df["time_left"] = hit_df["time_left"].apply(convertToMins)

def computeSimpleFeatures(hit_df, feature_df, dataset, list_sep):
    ## Basic features
    # (hits_completed_sd is a bust, the rest are salvageable)
    # NOTE: num_obs should NOT be imported for textlab, since this tells you the duration for
    # textlab_10 and textlab_30...
    # Also, including min_time_gap and max_time_gap for ipeirotis gives nearly perfect prediction :/
    # As does avg_time_gap and med_time_gap and time_gap_sd ://
    vars_to_import = ["first_hits","last_hits","avg_hitrate","avg_hits_completed",
                     "med_hits_completed","min_hits_completed","max_hits_completed",
                     "num_zeros"]
    if dataset != "ipeirotis":
        vars_to_import.append("time_left")
    if dataset == "ipeirotis":
        vars_to_import.append("num_obs")
    for cur_var in vars_to_import:
        feature_df[cur_var] = hit_df[cur_var]
        
    # Fill in the NaNs
    feature_df["avg_hits_completed"] = feature_df["avg_hits_completed"].fillna(0)
    feature_df["med_hits_completed"] = feature_df["med_hits_completed"].fillna(0)
    feature_df["min_hits_completed"] = feature_df["min_hits_completed"].fillna(0)
    feature_df["max_hits_completed"] = feature_df["max_hits_completed"].fillna(0)

    if dataset != "textlab_30":
        # No max_hits in textlab_30
        feature_df["max_hits"] = hit_df["max_hits"].astype(int)

    ## Now extract some lengths
    feature_df["title_len"] = hit_df["title"].apply(len)
    if dataset != "textlab_30":
        feature_df["desc_len"] = hit_df["description"].apply(len)
        feature_df["keywords_len"] = hit_df["keywords"].apply(lambda x: len(x) if pd.notnull(x) else 0)
        feature_df["num_keywords"] = (hit_df["keywords"].str.count(list_sep) + 1).fillna(0).astype(int)

    # Requester count in the dataset
    hit_df["req_count"] = 0
    feature_df["req_count"] = hit_df.groupby("requester_id")["req_count"].transform("count")

    ## Number of words
    feature_df["title_words"] = (hit_df["title"].str.split()).apply(len)
    if dataset != "textlab_30":
        feature_df["desc_words"] = (hit_df["description"].str.split()).apply(len)

    return feature_df

def computeIndicatorFeatures(hit_df, feature_df, dataset, indicative_map,
    gadiraju_categories, desc_top, title_top, kw_top):
    ## Indicative words: using lowercase versions of title+descriptions+keywords
    hit_df["title_lower"] = hit_df["title"].str.lower()
    if dataset != "textlab_30":
        hit_df["desc_lower"] = hit_df["description"].str.lower()
        ## Keyword is a bit different because sometimes it's NaN
        hit_df["kw_lower"] = hit_df["keywords"].str.lower()

    ## Use indicative words as a proxy for full ngramming
    ## [see gadiraju et al 2014, yang et al 2017, plus I made some up]
    for var_num, var_suffix in enumerate(indicative_map):
        print("Computing feature #" + str(var_num) + ": " + var_suffix)
        str_to_search = indicative_map[var_suffix]
        title_var = "title_" + var_suffix
        feature_df[title_var] = (hit_df["title_lower"].str.count(str_to_search)).astype(int)
        if dataset != "textlab_30":
            desc_var = "desc_" + var_suffix
            feature_df[desc_var] = (hit_df["desc_lower"].str.count(str_to_search)).astype(int)
            kw_var = "kw_" + var_suffix
            feature_df[kw_var] = (hit_df["kw_lower"].str.count(str_to_search).fillna(0)).astype(int)

    ## New as of 2018-06-19: Export a separate dataset with the Gadiraju
    ## categories for each HIT group
    #print(hit_df.columns)
    #print("****")
    #print(feature_df.columns)
    gadiraju_df = pd.DataFrame()
    for cur_category in gadiraju_categories:
        #print(cur_category)
        suffixes = gadiraju_categories[cur_category]
        # Titles
        title_var = "title_" + cur_category
        title_feature_vars = ["title_" + suffix for suffix in suffixes]
        title_subset_df = feature_df[title_feature_vars]
        gadiraju_df[title_var] = title_subset_df.sum(axis=1)
        # And now description and kw if not textlab_30
        if dataset != "textlab_30":
            # Descriptions
            desc_var = "desc_" + cur_category
            desc_feature_vars = ["desc_" + suffix for suffix in suffixes]
            desc_subset_df = feature_df[desc_feature_vars]
            gadiraju_df[desc_var] = desc_subset_df.sum(axis=1)
            # Keywords
            kw_var = "kw_" + cur_category
            kw_feature_vars = ["kw_" + suffix for suffix in suffixes]
            kw_subset_df = feature_df[kw_feature_vars]
            gadiraju_df[kw_var] = kw_subset_df.sum(axis=1)

    # Add group_id as the first column
    gadiraju_df.insert(0, "group_id", feature_df.index)
    # And export
    gadiraju_filepath = os.path.join(ml_input_path, "gadiraju_categories_" + dataset + ".csv")
    gadiraju_df.to_csv(gadiraju_filepath, index=False)

    ## Now the top 400 learned features (200 for rewards and 200 for duration)
    for var_suffix in title_top:
        str_to_search = title_top[var_suffix]
        title_var = "title_" + var_suffix.replace(" ","_")
        feature_df[title_var] = (hit_df["title_lower"].str.count(str_to_search)).astype(int)
    if dataset != "textlab_30":
        for var_suffix in desc_top:
            str_to_search = desc_top[var_suffix]
            desc_var = "desc_" + var_suffix.replace(" ","_")
            feature_df[desc_var] = (hit_df["desc_lower"].str.count(str_to_search)).astype(int)
        for var_suffix in kw_top:
            str_to_search = kw_top[var_suffix]
            kw_var = "kw_" + var_suffix.replace(" ","_")
            feature_df[kw_var] = (hit_df["kw_lower"].str.count(str_to_search).fillna(0)).astype(int)

    ## Other stuff
    ## 0/1 does a phrase like "5 minute task" appear
    seconds_reg = r'([0-9]+)\s?(?:s|sec|second|secs|seconds)'
    minutes_reg = r'([0-9]+)\s?(?:m|min|minute|mins|minutes)'
    hours_reg = r'([0-9]+)\s?(?h|hr|hour|hrs|hours)'
    feature_df["minutes_title"] = hit_df["title_lower"].str.extract(minutes_reg,expand=False).fillna(0).astype(int)
    if dataset != "textlab_30":
        feature_df["minutes_desc"] = hit_df["desc_lower"].str.extract(minutes_reg,expand=False).fillna(0).astype(int)
        feature_df["minutes_kw"] = hit_df["kw_lower"].str.extract(minutes_reg,expand=False).fillna(0).astype(int)

def computeQualificationFeatures(hit_df, feature_df, list_sep):
    # Qualification regexes
    qual_granted = r'\'.+ has been granted\''
    qual_not_granted = r'\'.+ has not been granted\''
    qual_loc = r'\'Location is (..)\''
    qual_loc_mult = r'\'Location is one of: (.+)\''
    qual_rej_rate_lt = r'\'HIT rejection rate \(%\) is less than ([0-9]+)\''
    qual_rej_num_lt = r'\'Total rejected HITs is (?:less than|not greater than) ([0-9]+)\''
    qual_appr_rate_gt = r'\'HIT approval rate \(%\) is (?:greater than|not less than) ([0-9]+)\''
    qual_appr_num_gt = r'\'Total approved HITs is (?:greater than|not less than) ([0-9]+)\''
    qual_adult_content = r'\'Adult Content Qualification is 1\''

    # Parse the qualifications list
    # Include the .fillna(val) at the end to determine the variable's value for
    # HITs without any qualifications
    feature_df["qual_len"] = hit_df["qualifications"].apply(lambda x: len(x) if pd.notnull(x) else 0)
    feature_df["num_quals"] = (hit_df["qualifications"].str.count(list_sep) + 1).fillna(0).astype(int)
    ## Custom quals
    feature_df["custom_not_granted"] = hit_df["qualifications"].str.count(qual_not_granted).fillna(0).astype(int)
    feature_df["custom_granted"] = hit_df["qualifications"].str.count(qual_granted).fillna(0).astype(int)

    ## Location qualifications
    # expand=False makes sure that data is squeezed into a Series not a DF
    hit_df["locs"] = hit_df["qualifications"].str.extract(qual_loc,expand=False).fillna('')
    hit_df["locs_mult"] = hit_df["qualifications"].str.extract(qual_loc_mult,expand=False).fillna('')
    # any_loc = 0 or 1, any location qualifications
    feature_df["any_loc"] = ((hit_df["locs"] + hit_df["locs_mult"]) != "").astype(int)
    # us_only = 0 or 1 if location qualification is US only
    feature_df["us_only"] = ((hit_df["locs"] + "|" + hit_df["locs_mult"]).str.contains("US")).astype(int)

    ## Rejection / acceptance qualifications
    # rej_rate_lt = restricted to workers with rejection pct less than X
    # (-1 = no restrictions, 0-100 = required rejection pct)
    feature_df["appr_rate_gt"] = hit_df["qualifications"].str.extract(qual_appr_rate_gt,expand=False).fillna(-1).astype(int)
    feature_df["appr_num_gt"] = hit_df["qualifications"].str.extract(qual_appr_num_gt,expand=False).fillna(-1).astype(int)

def mergeLDA(feature_df, dataset):
    # First, generate a temporary index for feature_df
    #feature_df.index = feature_df.index.astype(int)
    if "level_0" not in feature_df.columns:
        feature_df.reset_index(inplace=True)
    # Merge in topic distributions
    for K in [5,10,15,20]:
        title_topic_filename = os.path.join(lda_path,
            dataset + "_title_lda_dists_" + str(K) + ".pkl")
        title_topic_df = pd.read_pickle(title_topic_filename)
        feature_df = feature_df.merge(title_topic_df, how='left', left_index=True, right_index=True)
        if dataset != "textlab_30":
            desc_topic_filename = os.path.join(lda_path,
                dataset + "_desc_lda_dists_" + str(K) + ".pkl")
            desc_topic_df = pd.read_pickle(desc_topic_filename)
            feature_df = feature_df.merge(desc_topic_df, how='left', left_index=True, right_index=True)
            kw_topic_filename = os.path.join(lda_path,
                dataset + "_kw_lda_dists_" + str(K) + ".pkl")
            kw_topic_df = pd.read_pickle(kw_topic_filename)
            feature_df = feature_df.merge(kw_topic_df, how='left', left_index=True, right_index=True)
    print("LDA features merged in")

    ## New 2018-08-20: Output the topic distributions to a separate .csv file
    # If output_dists is True, output the feature_df with just the topic proportions to csv
    vars_to_keep = ["group_id","titletopic_5_0","titletopic_5_1",
        "titletopic_5_2","titletopic_5_3","titletopic_5_4"]
    if dataset != "textlab_30":
        vars_to_keep.extend(["desctopic_5_0","desctopic_5_1","desctopic_5_2",
            "desctopic_5_3","desctopic_5_4"])
        vars_to_keep.extend(["kwtopic_5_0","kwtopic_5_1","kwtopic_5_2",
            "kwtopic_5_3","kwtopic_5_4"])

    full_topic_df = feature_df[vars_to_keep]
    full_topic_filepath = os.path.join(ml_input_path, "topic_dists_" + dataset + ".csv")
    full_topic_df.to_csv(full_topic_filepath, index=False)

def mergeDoc2Vec(feature_df, dataset):
    # And doc2vec vectors
    doc2vec_df = pd.read_pickle(os.path.join(doc2vec_path, dataset + "_doc2vec.pkl"))
    feature_df = feature_df.merge(doc2vec_df, how='left', left_index=True, right_index=True)
    print("Doc2Vec features merged in")
    print(feature_df.tail())

def checkNulls(feature_df):
    ## Sanity check that none of the features have inf/nan values
    for cur_var in feature_df.columns:
        if cur_var == "group_id":
            continue
        #print("Checking " + cur_var)
        num_nans = sum(np.isnan(feature_df[cur_var]))
        num_inf = sum(np.isinf(feature_df[cur_var]))
        if num_nans > 0 or num_inf > 0:
            print(cur_var + ": " + str(num_nans) + " nans, " + str(num_inf) + " infs")

def fixNulls(feature_df):
    # avg_hitrate has 9 nans, 7 infs in ipeirotis data, and 8007 NaNs in textlab_30.
    # This gets fixed here by replacing nans with mean and infs with max
    non_inf = feature_df[~np.isinf(feature_df["avg_hitrate"])]
    mean_hitrate = non_inf["avg_hitrate"].mean()
    max_hitrate = non_inf["avg_hitrate"].max()
    print("Mean hitrate: " + str(mean_hitrate))
    print("Max hitrate: " + str(max_hitrate))
    feature_df["avg_hitrate"] = feature_df["avg_hitrate"].fillna(mean_hitrate)
    feature_df["avg_hitrate"] = feature_df["avg_hitrate"].replace(np.inf, max_hitrate)

#################################
### N-GRAM FEATURE EXTRACTION ###
#################################

def generateGramFeatures(hit_df, dataset, mode, partition):
    # For text ML, note that we have to do the training/validation split *right now*,
    # so that we don't fit the CountVectorizer on words that only appear in the
    # test set
    vars_to_keep = ["log_reward","log_duration","title","meandiff_lreward","meandiff_ldur"]
    if dataset == "ipeirotis":
        # TextLab scrapes don't have qualifications list, but Ipeirotis' data does
        vars_to_keep.append("qualifications")
    if dataset == "textlab_10" or dataset == "ipeirotis":
        # textlab_30 data doesn't have descriptions or keywords
        vars_to_keep.append("description")
        vars_to_keep.append("kw_parsed")

    feature_df = hit_df[vars_to_keep]

    if dataset == "textlab_10" or dataset == "ipeirotis":
        feature_df["description"] = feature_df["description"].astype(str)
        feature_df["kw_parsed"] = feature_df["kw_parsed"].astype(str)
    if dataset == "ipeirotis":
        feature_df["qualifications"] = feature_df["qualifications"].astype(str)

    # Make sure that the text vars are actually of *string* type.
    feature_df["title"] = feature_df["title"].astype(str)

    # Now, since we have the indices of the A/B split, and their train/validation
    # subsplits, separate these out into different dataframes *before* training CountVectorizer
    if mode.endswith("ab"):
        train_df = feature_df.iloc[partition['A_train']]
        test_df = feature_df.iloc[partition['A_val']]
    else:
        train_df = feature_df.iloc[partition['B_train']]
        test_df = feature_df.iloc[partition['B_val']]

    # Training Counts
    if dataset == "textlab_10" or dataset == "ipeirotis":
        ## Vectorize descriptions
        count_vec_desc = CountVectorizer(ngram_range=(1,3),stop_words='english',max_df=0.9)
        train_desc_counts = count_vec_desc.fit_transform(train_df["description"])
        #tfidf_vec_desc = TfidfVectorizer(ngram_range=(1,3),stop_words='english',max_df=0.9)
        #train_desc_tfidf = tfidf_vec_desc.fit_transform(train_df["description"])
        print("Descriptions vectorized for train data")
        ## Vectorize keywords (Unigrams only for the keywords strings)
        count_vec_kw = CountVectorizer(stop_words='english',max_df=0.9)
        train_kw_counts = count_vec_kw.fit_transform(train_df["kw_parsed"])
        #tfidf_vec_kw = TfidfVectorizer(stop_words='english',max_df=0.9)
        #train_kw_tfidf = tfidf_vec_kw.fit_transform(train_df["keywords"])
        print("Keywords vectorized for train data")
    ## Vectorize titles
    count_vec_title = CountVectorizer(ngram_range=(1,3),stop_words='english',max_df=0.9)
    train_title_counts = count_vec_title.fit_transform(train_df["title"])
    #tfidf_vec_title = TfidfVectorizer(ngram_range=(1,3),stop_words='english',max_df=0.9)
    #train_title_tfidf = tfidf_vec_title.fit_transform(train_df["title"])
    print("Titles vectorized for train data")

    if dataset == "textlab_10" or dataset == "ipeirotis":
        print(train_desc_counts.shape)
        print(train_kw_counts.shape)
    print(train_title_counts.shape)

    # Export feature names to file
    if dataset == "textlab_10" or dataset == "ipeirotis":
        desc_feature_name_arr = count_vec_desc.get_feature_names()
        desc_feature_name_arr = ["\"" + str(feat_name) + "\" in description" for feat_name in desc_feature_name_arr]
        kw_feature_name_arr = count_vec_kw.get_feature_names()
        kw_feature_name_arr = ["\"" + str(feat_name) + "\" in keywords" for feat_name in kw_feature_name_arr]
    title_feature_name_arr = count_vec_title.get_feature_names()
    title_feature_name_arr = ["\"" + str(feat_name) + "\" in title" for feat_name in title_feature_name_arr]

    feature_name_arr = title_feature_name_arr
    if dataset == "textlab_10" or dataset == "ipeirotis":
        feature_name_arr = feature_name_arr + desc_feature_name_arr + kw_feature_name_arr
    feature_filename = os.path.join(ml_input_path, "feat_names_" + dataset + "_" + mode + ".pkl")
    joblib.dump(feature_name_arr, feature_filename)
    print("Feature names saved to " + feature_filename)

    ## Validation counts
    if dataset == "textlab_10" or dataset == "ipeirotis":
        val_desc_counts = count_vec_desc.transform(test_df["description"])
        #test_desc_tfidf = tfidf_vec_desc.transform(test_df["description"])
        print("Validation set descriptions vectorized")
        val_kw_counts = count_vec_kw.transform(test_df["kw_parsed"])
        #test_kw_tfidf = tfidf_vec_kw.transform(test_df["keywords"])
        print("Validation set keywords vectorized")
    val_title_counts = count_vec_title.transform(test_df["title"])
    #test_title_tfidf = tfidf_vec_title.transform(test_df["title"])
    print("Validation set titles vectorized")

    if dataset == "textlab_10" or dataset == "ipeirotis":
        print(val_desc_counts.shape)
        print(val_kw_counts.shape)
    print(val_title_counts.shape)

    # Careful in this part:
    # train = A and test = B if [mode=="numab"]
    # train = A_train and test = A_validation if [mode=="textab"]
    # train = B and test = A if [mode=="numba"]
    # train = B_train and test = B_validation if [mode=="textba"]

    #####################
    ### TRAINING DATA ###
    #####################

    ### First we export the labels (the order is important because we drop the labels
    ### when exporting features in order to save memory - rather than making a new df)

    ## Training data: y1(rewards)
    train_labels_rew = train_df["log_reward"]
    train_rewards_filename = os.path.join(ml_input_path,
        "./train_rew_" + dataset + "_" + str(mode) + ".pkl")
    train_labels_rew.to_pickle(train_rewards_filename)
    print("Train rewards saved to " + train_rewards_filename)
    ## Training data: y2(duration)
    train_labels_dur = train_df["log_duration"]
    train_durations_filename = os.path.join(ml_input_path,
        "./train_dur_" + dataset + "_" + str(mode) + ".pkl")
    train_labels_dur.to_pickle(train_durations_filename)
    print("Train durations saved to " + train_durations_filename)

    ### Now we export the features
    ## Training data: text features
    train_text_features_filename = os.path.join(ml_input_path,
        "./train_txtfeats_" + dataset + "_" + str(mode) + ".pkl")
    if dataset == "textlab_10" or dataset == "ipeirotis":
        train_text_features = hstack([train_desc_counts,train_title_counts,train_kw_counts])
    else:
        train_text_features = train_title_counts
    joblib.dump(train_text_features, train_text_features_filename)
    print("Train text features saved to " + train_text_features_filename)
    print("Training data saved")

    #################
    ### TEST DATA ###
    #################

    ## Test data: y1(rewards)
    test_labels_rew = test_df["log_reward"]
    test_rewards_filename = os.path.join(ml_input_path,
        "./test_rew_" + dataset + "_" + str(mode) + ".pkl")
    test_labels_rew.to_pickle(test_rewards_filename)
    print("Test rewards saved to " + test_rewards_filename)
    ## Test data: y2(duration)
    test_labels_dur = test_df["log_duration"]
    test_durations_filename = os.path.join(ml_input_path,
        "./test_dur_" + dataset + "_" + str(mode) + ".pkl")
    test_labels_dur.to_pickle(test_durations_filename)
    print("Test durations saved to " + test_durations_filename)

    ## Validation data: text features
    test_text_features_filename = os.path.join(ml_input_path,
        "./test_txtfeats_" + dataset + "_" + str(mode) + ".pkl")
    if dataset == "textlab_10" or dataset == "ipeirotis":
        val_text_features = hstack([val_desc_counts,val_title_counts,val_kw_counts])
    else:
        val_text_features = val_title_counts
    joblib.dump(val_text_features, test_text_features_filename)
    print("Test text features saved to " + test_text_features_filename)

    print("Validation data saved")

    print("Textual features exported")
    print("*** Train data: ***")
    print("Features: " + str(train_text_features.shape))
    print("Reward labels: " + str(train_labels_rew.shape))
    print("Duration labels: " + str(train_labels_dur.shape))
    print("*** Validation data: ***")
    print("Features: " + str(val_text_features.shape))
    print("Reward labels: " + str(test_labels_rew.shape))
    print("Duration labels: " + str(test_labels_dur.shape))

def prepareML(dataset, mode):
    # Make sure the ml input path exists
    if not os.path.isdir(ml_input_path):
        os.mkdir(ml_input_path)
    cleaned_df = loadCleanedData(dataset)
    # Get the indices for each train/validation/test set
    # data_partition keys = {A, B, A_train, A_val, B_train, B_val} 
    data_partition = partitionData(cleaned_df)
    if mode.startswith("full"):
        generateNumericFeatures(cleaned_df, dataset, mode, data_partition)
    else:
        generateGramFeatures(cleaned_df, dataset, mode, data_partition)

def main():
    # Parses command-line arguments before calling prepareML()
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("mode")
    args = parser.parse_args()
    
    prepareML(args.dataset, args.mode)

if __name__ == "__main__":
    main()