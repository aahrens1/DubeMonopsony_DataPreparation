# Computes the summary stats for each HIT group in the panel: max hits, 
# total_hits (the sum of all the increases we observe in the time series for a
# given group), first hits, last hits, plus time gap features. Then collapses
# into a group-level dataset and merges in the metadata ([dataset]_meta.csv),
# to produce the final ([dataset]_group.csv) dataset
import os

import pandas as pd
import numpy as np

data_path = os.path.join("..","raw_data")
group_path = os.path.join("..","group_data")

def computeGroupStats(dataset):
    # Before anything else, make sure there's a group data (output) folder
    if not os.path.isdir(group_path):
        os.mkdir(group_path)
    
    panel_filename = os.path.join(data_path, dataset + "_panel.csv")
    hits_panel = pd.read_csv(panel_filename)

    # Rename the textlab vars
    if dataset == "textlab_10" or dataset == "textlab_30":
        hits_panel.drop(columns="Unnamed: 0",inplace=True)
        hits_panel.rename({"hit_series":"hits_available","ts_parsed":"timestamp"},axis="columns",inplace=True)

    # Convert timestamp to actual datetime objects
    if dataset == "ipeirotis":
        # old format
        #panel_df["timestamp"] = pd.to_datetime(panel_df["timestamp"],format="%Y-%m-%d_%H.%M")
        # new format
        hits_panel["timestamp"] = pd.to_datetime(hits_panel["timestamp"])
    elif dataset == "textlab_10" or dataset == "textlab_30":
        print("Parsing textlab dates")
        hits_panel["timestamp"] = pd.to_datetime(hits_panel["timestamp"])

    # Here we make two "fake" obs for each group - one n mins before the first real obs and one
    # n mins after. So that we won't have missing feature values for the groups with only
    # one obs. NOTE: only for textlab data (there's no regular scrape frequency for Panos data)
    if dataset == "textlab_10":
        n_mins = pd.Timedelta(minutes=10)
    elif dataset == "textlab_30":
        n_mins = pd.Timedelta(minutes=30)
    def addFakeObs(df):
        # First off, this is only necessary if there's no variation in hits_available
        min_hits = df["hits_available"].min()
        max_hits = df["hits_available"].max()
        no_variation = max_hits == min_hits
        # And if the last obs has less than 5 HITs
        last_hits = df["hits_available"].iloc[-1]
        if no_variation and last_hits < 5:
            # First get the group_id
            group_id = df["group_id"].iloc[0]
            # Now construct the fake first and last timestamp
            fake_first = df["timestamp"].iloc[0] - n_mins
            fake_last = df["timestamp"].iloc[-1] + n_mins
            # Now make the fake first+last obs
            fake_first_df = pd.DataFrame.from_dict({'group_id':[group_id],'hits_available':[0],'timestamp':[fake_first]})
            fake_last_df = pd.DataFrame.from_dict({'group_id':[group_id],'hits_available':[0],'timestamp':[fake_last]})
            # And add them to beginning + end
            return pd.concat([fake_first_df, df, fake_last_df], ignore_index=True)
        else:
            return df
    if dataset == "textlab_10" or dataset == "textlab_30":
        ## Adds the fake obs group-by-group
        hits_panel = hits_panel.groupby("group_id").apply(addFakeObs).reset_index(level=0,drop=True).reset_index(drop=True)

    # First we make a dataset with the *differences* in hits_available and
    # timestamp between successive rows, then we zero out all the negative
    # values, and sum this by group_id to get the total number of HITs added to
    # the HIT group over its lifespan

    # Now compute the num hits added. Based on:
    # https://stackoverflow.com/questions/20648346/computing-diffs-within-groups-of-a-dataframe
    hits_panel.sort_values(by=['group_id','timestamp'],inplace=True)
    hits_panel["hit_diffs"] = hits_panel["hits_available"].diff()
    hits_panel["time_diffs"] = hits_panel["timestamp"].diff()
    # Now NaN out all *positive* HIT diffs (since we're interested only in completion rate,
    # whereas a positive diff means that HITs were *added* over that timespan)
    hits_panel["hits_completed"] = hits_panel["hit_diffs"].apply(lambda x: -x if x < 0 else np.nan)
    # And NaN out the first obs in each group
    mask = hits_panel["group_id"] != hits_panel["group_id"].shift(1)
    hits_panel['hits_completed'][mask] = np.nan
    hits_panel['time_diffs'][mask] = np.nan
    # Make an is_zero var for the num_zeros collapsed var
    hits_panel["is_zero"] = hits_panel["hits_available"].apply(lambda x: int(x == 0.0))
    # Make a time_diff var in hours
    hits_panel["time_diff_hrs"] = hits_panel["time_diffs"] / pd.Timedelta(hours=1)
    # And in minutes
    hits_panel["time_diff_mins"] = hits_panel["time_diffs"] / pd.Timedelta(minutes=1)
    # And a hits_per_hr rate
    hits_panel["hits_per_hr"] = hits_panel["hits_completed"] / hits_panel["time_diff_hrs"]

    # Now we group by group_id so we can sum

    hit_groups = hits_panel.groupby("group_id")
    ### compute group-level features
    group_features = []
    # max_hits
    max_df = hit_groups["hits_available"].max().rename("max_hits")
    group_features.append(max_df)
    # first_hits
    first_df = hit_groups["hits_available"].first().rename("first_hits")
    group_features.append(first_df)
    # last_hits
    last_df = hit_groups["hits_available"].last().rename("last_hits")
    group_features.append(last_df)
    # num_obs
    numobs_df = hit_groups["hits_available"].size().rename("num_obs")
    group_features.append(numobs_df)

    ### Now the diff-based HIT features
    ## OLD: total_hits
    #total_df = hit_groups["diffs"].sum().rename("total_hits")
    ## NEW: avg_hitrate
    avg_hitrate_df = hit_groups["hits_per_hr"].mean().rename("avg_hitrate")
    noninf_max = avg_hitrate_df[~np.isinf(avg_hitrate_df)].max()
    avg_hitrate_df.replace(np.inf, noninf_max, inplace=True)
    group_features.append(avg_hitrate_df)
    # Average number of HITs completed between obs
    avg_hits_completed = hit_groups["hits_completed"].mean().rename("avg_hits_completed")
    group_features.append(avg_hits_completed)
    # Median number of HITs completed betweeen obs
    med_hits_completed = hit_groups["hits_completed"].median().rename("med_hits_completed")
    group_features.append(med_hits_completed)
    # Min HITs completed
    min_hits_completed = hit_groups["hits_completed"].min().rename("min_hits_completed")
    group_features.append(min_hits_completed)
    # Max HITs completed
    max_hits_completed = hit_groups["hits_completed"].max().rename("max_hits_completed")
    group_features.append(max_hits_completed)
    # Standard deviation of HITs completed
    hits_completed_sd = hit_groups["hits_completed"].std().rename("hits_completed_sd")
    group_features.append(hits_completed_sd)

    # first_time
    firsttime_df = hit_groups["timestamp"].first().rename("first_time")
    group_features.append(firsttime_df)
    # last_time
    lasttime_df = hit_groups["timestamp"].last().rename("last_time")
    group_features.append(lasttime_df)

    ### Now the diff-based time features
    # Average gap between obs
    avg_time_gap = hit_groups["time_diff_mins"].mean().rename("avg_time_gap")
    group_features.append(avg_time_gap)
    # Median time gap
    med_time_gap = hit_groups["time_diff_mins"].median().rename("med_time_gap")
    group_features.append(med_time_gap)
    # Min time gap
    min_time_gap = hit_groups["time_diff_mins"].min().rename("min_time_gap")
    group_features.append(min_time_gap)
    # Max time gap
    max_time_gap = hit_groups["time_diff_mins"].max().rename("max_time_gap")
    group_features.append(max_time_gap)
    # time gap standard deviation
    time_gap_sd = hit_groups["time_diff_mins"].std().rename("time_gap_sd")
    group_features.append(time_gap_sd)

    # num_zeros
    numzeros_df = hit_groups["is_zero"].sum().rename("num_zeros")
    group_features.append(numzeros_df)

    # Now bring them all together into a HIT *group*-level dataset
    group_level = pd.concat(group_features,axis=1)

    # And compute the durations
    group_level["duration"] = group_level["last_time"] - group_level["first_time"]

    # Convert to minutes
    #group_level["duration"] = group_level["duration"].astype('timedelta64[m]')
    group_level["duration"] = group_level["duration"].dt.total_seconds() / 60.0

    group_filename = os.path.join(group_path, dataset + "_group_stats.pkl")
    group_level.to_pickle(group_filename)

    csv_filename = os.path.join(group_path, dataset + "_group_stats.csv")
    group_level.to_csv(csv_filename)

    # Now merge these stats back in with the titles, descriptions, etc. from the
    # metadata file

    meta_filename = os.path.join(data_path, dataset + "_meta.csv")
    requester_data = pd.read_csv(meta_filename)

    if dataset == "textlab_10":
        requester_vars = ["group_id","reward","time_left","requester","time_allotted","title",
                            "keywords","requester_id","description","expiration_date"]
    elif dataset == "textlab_30":
        requester_vars = ["group_id","requester_id","title","requester","expiration_date",
                         "reward","time_allotted","time_left"]
    elif dataset == "ipeirotis":
        requester_vars = ["group_id","requester_id","title","description","keywords",
                         "time_allotted","reward","qualifications","expiration_date"]
    requester_data = requester_data[requester_vars]

    # Sanity check
    print("requester data length: " + str(len(requester_data)))
    print("unique group_ids in requester data: " + str(len(requester_data["group_id"].value_counts())))

    requester_data.set_index("group_id",inplace=True)
    # Now merge in the group stats from above
    merged_df = group_level.merge(requester_data, left_index=True, right_index=True, indicator=True)

    # Chek that everything merged correctly
    num_badmerge = sum(merged_df["_merge"] != "both")

    if num_badmerge > 0:
        print("Bad merge!!")
        quit()

    # If 0, drop the _merge var
    merged_df.drop(columns="_merge",inplace=True)

    csv_filename = os.path.join(group_path, dataset + "_group.csv")
    merged_df.to_csv(csv_filename)

    pkl_filename = os.path.join(group_path, dataset + "_group.pkl")
    merged_df.to_pickle(pkl_filename)


def main():
    # Parses command-line arguments before calling the "workhorse" function
    # computeGroupStats()
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    computeGroupStats(args.dataset)

if __name__ == "__main__":
    main()