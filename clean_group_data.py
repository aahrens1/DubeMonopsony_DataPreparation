# Winsorizes the data, removes invalid observations, and gets the variables
# into the data format needed for the remainder of the pipeline
import argparse
import datetime
import os
import re

import pandas as pd
import numpy as np

group_data_path = os.path.join("..","group_data")
cleaned_path = os.path.join("..","cleaned_data")

def cleanGroupData(dataset):
    # Before anything else, make sure there's a cleaned data (output) folder
    if not os.path.isdir(cleaned_path):
        os.mkdir(cleaned_path)

    # Load the group-level non-cleaned dataset
    if dataset == "ipeirotis":
        ## Load the data from Ipeirotis API
        df_filepath = os.path.join(group_data_path,"ipeirotis_group.pkl")
        hit_df = pd.read_pickle(df_filepath)
        # (once the data-cleaning cells below have been run once, you can
        # skip down to Step 2.1 and just load e.g. ipeirotis_cleaned.pkl)
    elif dataset == "textlab_30":
        ## Load the data from the TextLab 30-minute scrapes
        df_filepath = os.path.join(group_data_path,"textlab_30_group.pkl")
        hit_df = pd.read_pickle(df_filepath)
    else:
        ## Load the data from the TextLab 10-minute scrapes. Note that this is the data
        ## *produced by Compute_Group_Stats.ipynb*, which merges the panel with the metadata
        ## and then collapses into a group-level dataset [loaded here]
        df_filepath = os.path.join(group_data_path,"textlab_10_group.pkl")
        hit_df = pd.read_pickle(df_filepath)
    print("Loaded " + dataset + ": " + str(len(hit_df)) + " observations")

    if dataset == "textlab_10" or dataset == "textlab_30":
        ## First, convert the reward from string to int
        hit_df["reward"] = hit_df["reward"].str.replace("$","").astype(float) * 100
    ## compute log_reward and log_duration
    hit_df["log_reward"] = hit_df["reward"].apply(np.log)
    hit_df["log_duration"] = hit_df["duration"].apply(np.log)
    ## And mark as invalid those observations where last_hits > first_hits
    ## (Note: only for ipeirotis data)
    if dataset == "ipeirotis":
        hit_df["valid_duration"] = hit_df["last_hits"] <= hit_df["first_hits"]
    else:
        hit_df["valid_duration"] = 1
    print("Invalid durations marked")

    # Drop null rewards/durations
    print("# null log_reward: " + str(len(np.where(hit_df["log_reward"].isnull())[0])))
    print("# null log_duration: " + str(len(np.where(hit_df["log_duration"].isnull())[0])))
    print("Total number of obs: " + str(len(hit_df)))
    hit_df["log_duration"] = hit_df["log_duration"].replace(-np.inf, np.nan)
    hit_df["log_reward"] = hit_df["log_reward"].replace(-np.inf, np.nan)
    hit_df = hit_df.dropna(subset=["log_duration"])
    hit_df = hit_df.dropna(subset=["log_reward"])
    # Number of observations after invalid obs dropped
    print("New # observations after dropping null rewards/durations: " + str(len(hit_df)))

    ## Drop invalid durations
    hit_df = hit_df[hit_df["valid_duration"] == 1]
    hit_df = hit_df[hit_df["duration"] > 0]
    print("Number of HIT batches once invalid durations are dropped: " + str(len(hit_df.index)))
    ## Drop invalid rewards
    hit_df = hit_df[~hit_df["log_reward"].isnull()]
    print("And once invalid rewards are dropped: " + str(len(hit_df.index)))

    # Filter the outliers
    #rew_upper_cutoff = hit_df["reward"].quantile(0.995)
    rew_upper_cutoff = 500.0
    #dur_upper_cutoff = hit_df["duration_new"].quantile(0.995)
    dur_upper_cutoff = 90000.0
    print("Dropping obs with reward < 0 or reward > " + str(rew_upper_cutoff))
    print("Dropping obs with duration_new < 0 or duration_new > " + str(dur_upper_cutoff))
    hit_df = hit_df[(hit_df["duration"] > 0) & (hit_df["duration"] < dur_upper_cutoff)]
    hit_df = hit_df[(hit_df["reward"] > 0) & (hit_df["reward"] < rew_upper_cutoff)]
    print("Number of obs remaining: " + str(len(hit_df.index)))

    ## Normalized duration variable: duration_new / max_hits
    #hit_df["normduration_new"] = hit_df["duration_new"] / hit_df["max_hits"]
    #hit_df["log_normduration_new"] = hit_df["normduration_new"].apply(math.log)

    ## Description and title can be used as-is, but keywords need to be parsed for ipeirotis+textlab_10
    if dataset == "ipeirotis":
        keyword_reg = r'\'(\S+)\''
        hit_df["kw_parsed"] = hit_df["keywords"].apply(lambda x: ' '.join(re.findall(keyword_reg, x)) if pd.notnull(x) else '')
    elif dataset == "textlab_10":
        hit_df["kw_parsed"] = hit_df["keywords"].str.replace('|',' ')
        hit_df["kw_parsed"] = hit_df["kw_parsed"].str.replace('nan','')
        hit_df["kw_parsed"] = hit_df["kw_parsed"].astype(str)
        
    ## New: drop observations in Ipeirotis data occurring on or after February 1, 2016
    if dataset == "ipeirotis":
        feb1 = datetime.datetime(2016,2,1)
        hit_df = hit_df[hit_df["last_time"] < feb1]
        print("Number of obs after dropping Feb 1 onwards: " + str(len(hit_df)))

    ## Convert times from string format (e.g., "30 minutes") to int
    # (can't be too careful)
    min_reg = r'([0-9]+) (?:minute)|(?:minutes)'
    hr_reg = r'([0-9]+) (?:hour)|(?:hours)'
    day_reg = r'([0-9]+) (?:day)|(?:days)'
    week_reg = r'([0-9]+) (?:week)|(?:weeks)'

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

    ## Here we save the dataset as .pkl so that the LDA+doc2vec code can use the cleaned version
    cleaned_filepath = os.path.join(cleaned_path, dataset + "_cleaned.pkl")
    hit_df.to_pickle(cleaned_filepath)
    print("Cleaned dataset saved as " + str(cleaned_filepath))

def main():
    # Parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    cleanGroupData(args.dataset)

if __name__ == "__main__":
    main()