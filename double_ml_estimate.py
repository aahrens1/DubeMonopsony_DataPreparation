import argparse
import joblib
import os
import pandas as pd
import numpy as np

from global_settings import output_paths

cleaned_path = output_paths["cleaned"]
estimates_path = output_paths["estimates"]
ml_input_path = output_paths["ml_input"]
ml_output_path = output_paths["ml_output"]

# Final step! Compute the double ML estimate and output residuals to file

def computeThetaCheck(lrew_resid, ldur_resid):
    inv_term = 1/(sum(lrew_resid**2)/len(lrew_resid))
    numer_term = sum(lrew_resid * ldur_resid)/len(lrew_resid)
    theta_check = inv_term * numer_term
    return theta_check

def computeEta(dataset):

    ########################
    ### Part I: Estimate ###
    ########################

    print("doubleMLEstimate()")
    # Takes the predicted y values of the two ML runs, computes residuals, and
    # plugs them into the Double ML estimator formula

    # (LOTS OF STUFF IN THIS SECTION WILL LOOK WEIRD BECAUSE OF THE CHANGE FROM TRAIN/TEST to A/B)

    # (Set to True if you're also doing the alternative specification with
    # mean-differenced rewards and durations)
    mean_differenced = False

    ## Load best reward predictions
    predicted_lrew_ab = joblib.load(os.path.join(ml_output_path,
        "predictions_" + dataset + "_fullab_rew.pkl"))
    print("Loaded predicted_lrew_ab: " + str(predicted_lrew_ab.shape))
    predicted_lrew_ba = joblib.load(os.path.join(ml_output_path,
        "predictions_" + dataset + "_fullba_rew.pkl"))
    print("Loaded predicted_lrew_ba: " + str(predicted_lrew_ba.shape))
    ## Load best mdreward predictions
    #predicted_mdlrew = joblib.load("./predictions_rfr_mdlrew.pkl")
    #predicted_mdlrew_flip = joblib.load("./predictions_rfr_mdlrewflip.pkl")
    ## Load actual rewards
    lrew_ab = pd.read_pickle(os.path.join(ml_input_path,
        "train_rew_" + dataset + "_fullab.pkl"))
    print("Loaded lrew_ab: " + str(lrew_ab.shape))
    lrew_ba = pd.read_pickle(os.path.join(ml_input_path,
        "train_rew_" + dataset + "_fullba.pkl"))
    print("Loaded lrew_ba: " + str(lrew_ba.shape))
    ## Load actual mean-differenced rewards
    #mdlrew_train = pd.read_pickle("./train_mdrewards.pkl")
    #mdlrew_test = pd.read_pickle("./test_mdrewards.pkl")

    ## Load best duration predictions
    predicted_ldur_ab = joblib.load(os.path.join(ml_output_path,
        "predictions_" + dataset + "_fullab_dur.pkl"))
    predicted_ldur_ba = joblib.load(os.path.join(ml_output_path,
        "predictions_" + dataset + "_fullba_dur.pkl"))
    ## Load best mdlduration predictions
    #predicted_mdldur = joblib.load("./predictions_rfr_mdldur.pkl")
    #predicted_mdldur_flip = joblib.load("./predictions_rfr_mdldurflip.pkl")
    ## Load best normduration predictions
    #predicted_lndur = joblib.load("./predictions_rfr_normdur.pkl")
    ## Load actual durations
    ldur_ab = pd.read_pickle(os.path.join(ml_input_path,
        "train_dur_" + dataset + "_fullab.pkl"))
    ldur_ba = pd.read_pickle(os.path.join(ml_input_path,
        "train_dur_" + dataset + "_fullba.pkl"))
    ## Load actual mdldurations
    #mdldur_train = pd.read_pickle("./train_mdldurs.pkl")
    #mdldur_test = pd.read_pickle("./test_mdldurs.pkl")
    ## Load actual normdurations
    #lndur = pd.read_pickle("./test_normdurations.pkl")

    ## Compute reward residuals = V_hat
    # This part looks weird, but since the most-recently-predicted set was Set A
    # (the original training set), the Set A residuals are lrew_train - predicted_lrew
    # and vice-versa for Set B residuals
    lrew_resid_ab = lrew_ba - predicted_lrew_ab
    lrew_resid_ab.rename("lrew_resid",inplace=True)
    lrew_resid_ba = lrew_ab - predicted_lrew_ba
    lrew_resid_ba.rename("lrew_resid_ba",inplace=True)
    ## Compute mean-differenced log(reward) residuals
    #mdlrew_resid = mdlrew_test - predicted_mdlrew
    #mdlrew_resid.rename("mdlrew_resid",inplace=True)
    #mdlrew_resid_flip = mdlrew_train - predicted_mdlrew_flip
    #mdlrew_resid_flip.rename("mdlrew_resid_flip",inplace=True)

    ## Compute log(duration) residuals = W_hat
    ldur_resid_ab = ldur_ba - predicted_ldur_ab
    ldur_resid_ab.rename("ldur_resid",inplace=True)
    ldur_resid_ba = ldur_ab - predicted_ldur_ba
    ldur_resid_ba.rename("ldur_resid_ba",inplace=True)
    ## Compute mean-differenced log(duration) residuals
    #mdldur_resid = mdldur_test - predicted_mdldur
    #mdldur_resid.rename("mdldur_resid",inplace=True)
    #mdldur_resid_flip = mdldur_train - predicted_mdldur_flip
    #mdldur_resid_flip.rename("mdldur_resid_flip",inplace=True)
    ## Compute log(normalized duration) residuals
    #lndur_resid = lndur - predicted_lndur
    #lndur_resid.rename("lndur_resid",inplace=True)

    ## Theta_check(log(reward) -> log(duration))
    theta_check_ab = computeThetaCheck(lrew_resid_ab, ldur_resid_ab)
    print("Standard theta_check: " + str(theta_check_ab))
    theta_check_ba = computeThetaCheck(lrew_resid_ba, ldur_resid_ba)
    print("Standard theta_check (flipped data): " + str(theta_check_ba))
    avg_theta_check = (theta_check_ab + theta_check_ba)/2
    print("Averaged theta_check: " + str(avg_theta_check))

    ## Theta_check(md(log(reward)) -> md(log(duration)))
    #theta_check_md = computeThetaCheck(mdlrew_resid, mdldur_resid)
    #print("Mean-differenced theta_check: " + str(theta_check_md))
    #theta_check_md_flip = computeThetaCheck(mdlrew_resid_flip, mdldur_resid_flip)
    #print("Mean-differenced theta_check (flipped data): " + str(theta_check_md_flip))
    #avg_theta_check_md = (theta_check_md + theta_check_md_flip) / 2
    #print("Averaged mean-differenced theta_check: " + str(avg_theta_check_md))

    # Theta_check(log(reward) -> log(normduration))
    #theta_check_ndur = computeThetaCheck(lrew_resid, lndur_resid)
    #print("Standard theta_check with normalized durations: " + str(theta_check_ndur))

    # Theta_check(md(log(reward)) -> log(normduration))
    #theta_check_md_ndur = computeThetaCheck(mdlrew_resid, lndur_resid)
    #print("Mean-differenced theta_check with normalized durations: " + str(theta_check_md_ndur))

    ###############################
    ### Part II: Export to file ###
    ###############################

    ## Now that double ML estimates have been computed, merge the predicted values
    ## and residuals back into the original dataset
    residual_vars = [lrew_resid_ab, ldur_resid_ab]
    residual_vars_flip = [lrew_resid_ba,ldur_resid_ba]
    if mean_differenced:
        residual_vars = residual_vars + [mdlrew_resid, mdldur_resid]
        #residual_vars_flip = residual_vars_flip + [mdlrew_resid_flip, mdldur_resid_flip]
    residual_df = pd.concat(residual_vars, axis=1)
    residual_df_flip = pd.concat(residual_vars_flip, axis=1)
    #residual_df = pd.concat([mdlrew_resid, mdldur_resid], axis=1)

    # Merge predicted vals back into the test data DFs
    residual_df["predicted_lrew"] = predicted_lrew_ab
    residual_df_flip["predicted_lrew_ba"] = predicted_lrew_ba
    if mean_differenced:
        residual_df["predicted_mdlrew"] = predicted_mdlrew
        #residual_df_flip["predicted_mdlrew_flip"] = predicted_mdlrew_flip
    residual_df["predicted_ldur"] = predicted_ldur_ab
    residual_df_flip["predicted_ldur_ba"] = predicted_ldur_ba
    if mean_differenced:
        residual_df["predicted_mdldur"] = predicted_mdldur
        residual_df_flip["predicted_mdldur_flip"] = predicted_mdldur_flip
    #residual_df["predicted_lndur"] = predicted_lndur

    if dataset == "ipeirotis":
        ## Now merge *this* residual_df back into the full dataset
        full_df = pd.read_pickle(os.path.join(cleaned_path,"ipeirotis_cleaned.pkl"))
    if dataset == "textlab_30":
        ## Or the full 400k obs TextLab 30-minute dataset
        full_df = pd.read_pickle(os.path.join(cleaned_path,"textlab_30_cleaned.pkl"))
    if dataset == "textlab_10":
        ## Or 100k obs TextLab 10-minute dataset
        full_df = pd.read_pickle(os.path.join(cleaned_path,"textlab_10_cleaned.pkl"))
    # Turns out we have to drop desc+kw vars for Pandas to be able to write to .dta
    if dataset != "textlab_30":
        full_df.drop("description",axis=1,inplace=True)
        full_df.drop("keywords",axis=1,inplace=True)
        full_df.drop("kw_parsed",axis=1,inplace=True)
    #full_df.drop("qualifications",axis=1,inplace=True)
    # And reset the index, so it's 0, 1, 2, ... instead of the group ids
    #full_df.index = full_df.index.astype(int)
    full_df.reset_index(inplace=True)

    # Merge the "normal" training-test split residuals in
    full_df_merged = full_df.merge(residual_df,how='left',left_index=True,right_index=True,indicator=True)
    full_df_merged["_merge"] = full_df_merged["_merge"].replace('both',1)
    full_df_merged["_merge"] = full_df_merged["_merge"].replace('left_only',0)
    full_df_merged.rename(index=str,columns={'_merge':'A_B'},inplace=True)
    full_df_merged["A_B"] = full_df_merged["A_B"].astype(int)
    full_df_merged.index = full_df_merged.index.astype(int)

    flip_vars = ["lrew_resid_ba","ldur_resid_ba","predicted_lrew_ba",
                 "predicted_ldur_ba"]
    if mean_differenced:
        flip_vars = flip_vars + ["mdlrew_resid_flip","mdldur_resid_flip",
                                "predicted_mdlrew_flip","predicted_mdldur_flip"]

    # And merge the flipped residuals in
    full_df_merged = full_df_merged.merge(residual_df_flip,how='left',left_index=True,right_index=True,indicator=True)
    full_df_merged["_merge"] = full_df_merged["_merge"].replace('both',2)
    full_df_merged["_merge"] = full_df_merged["_merge"].replace('left_only',0)
    full_df_merged["_merge"] = full_df_merged["_merge"].astype(int)
    full_df_merged["A_B"] = full_df_merged["A_B"] + full_df_merged["_merge"]
    full_df_merged.drop("_merge",axis=1,inplace=True)

    # Now fill the (non-flipped) NaN values with the flipped values
    full_df_merged["lrew_resid"] = full_df_merged["lrew_resid"].fillna(full_df_merged["lrew_resid_ba"])
    #full_df_merged["mdlrew_resid"] = full_df_merged["mdlrew_resid"].fillna(full_df_merged["mdlrew_resid_flip"])
    full_df_merged["ldur_resid"] = full_df_merged["ldur_resid"].fillna(full_df_merged["ldur_resid_ba"])
    #full_df_merged["mdldur_resid"] = full_df_merged["mdldur_resid"].fillna(full_df_merged["mdldur_resid_flip"])
    full_df_merged["predicted_lrew"] = full_df_merged["predicted_lrew"].fillna(full_df_merged["predicted_lrew_ba"])
    #full_df_merged["predicted_mdlrew"] = full_df_merged["predicted_mdlrew"].fillna(full_df_merged["predicted_mdlrew_flip"])
    full_df_merged["predicted_ldur"] = full_df_merged["predicted_ldur"].fillna(full_df_merged["predicted_ldur_ba"])
    #full_df_merged["predicted_mdldur"] = full_df_merged["predicted_mdldur"].fillna(full_df_merged["predicted_mdldur_flip"])

    full_df_merged.drop(flip_vars,axis=1,inplace=True)

    #####################
    ### The export!!! ###
    #####################

    # Load the R^2 terms from the ML runs
    format_str = "{0:.4f}"
    lrew_r2 = format_str.format(joblib.load(os.path.join(ml_output_path,
        "pred_score_" + dataset + "_fullab_rew.pkl")))
    lrew_r2_flip = format_str.format(joblib.load(os.path.join(ml_output_path,
        "pred_score_" + dataset + "_fullba_rew.pkl")))
    ldur_r2 = format_str.format(joblib.load(os.path.join(ml_output_path,
        "pred_score_" + dataset + "_fullab_dur.pkl")))
    ldur_r2_flip = format_str.format(joblib.load(os.path.join(ml_output_path,
        "pred_score_" + dataset + "_fullba_dur.pkl")))
    if mean_differenced:
        mdlrew_r2 = format_str.format(joblib.load(os.path.join(ml_output_path,
            "pred_score_rfr_mdlrew.pkl")))
        #mdlrew_r2_flip = format_str.format(joblib.load("pred_score_rfr_mdlrewflip.pkl"))
        mdldur_r2 = format_str.format(joblib.load(os.path.join(ml_output_path,
            "pred_score_rfr_mdldur.pkl")))
        #mdldur_r2_flip = format_str.format(joblib.load("pred_score_rfr_mdldurflip.pkl"))

    # Export dataset of residuals to Stata
    label_map = {}
    label_map["lrew_resid"] = "log(reward) residuals. Rsq_0 = " + lrew_r2 + ", Rsq_1 = " + lrew_r2_flip
    label_map["ldur_resid"] = "log(duration) residuals. Rsq_0 = " + ldur_r2 + ", Rsq_1 = " + ldur_r2_flip
    if mean_differenced:
        label_map["mdlrew_resid"] = "Mean-differenced log(reward) residuals. Rsq_0 = " + mdlrew_r2 + ", Rsq_1 = " + mdlrew_r2_flip
        label_map["mdldur_resid"] = "Mean-differenced log(duration) residuals. Rsq_0 = " + mdldur_r2 + ", Rsq_1 = " + mdldur_r2_flip
    label_map["A_B"] = "0 if not used, 1 if in Set A (first train data), 2 if in Set B (first test data)"
    residuals_filepath = os.path.join(estimates_path, "residuals_full_" + dataset + ".dta")
    if "level_0" in full_df_merged.columns:
        full_df_merged.drop("level_0",1,inplace=True)
    if "qualifications" in full_df_merged.columns:
        full_df_merged.drop("qualifications",1,inplace=True)
    full_df_merged.to_stata(residuals_filepath, variable_labels=label_map)
    print("Exported Stata file " + residuals_filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    computeEta(args.dataset)

if __name__ == '__main__':
    main()