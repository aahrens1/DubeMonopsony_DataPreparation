import os
import csv
import joblib

from global_settings import output_paths

datasets_chronological = ["ipeirotis", "textlab_30", "textlab_10"]

# Load the ML results from ml_output folder
ml_output_path = output_paths["ml_output"]
# Save R2 table in the estimates folder
estimates_path = output_paths["estimates"]

def generateR2Table():
    table_rows = [["Reward"],["Duration"]]
    for dataset in datasets_chronological:
        for mode in ["fullab","fullba"]:
            rew_filepath = os.path.join(ml_output_path,
              "pred_score_" + dataset + "_" + mode + "_rew.pkl")
            if os.path.isfile(rew_filepath):
                rew_r2 = '{:.4f}'.format(round(joblib.load(rew_filepath),4))
            else:
                rew_r2 = ""
            dur_filepath = os.path.join(ml_output_path,
              "pred_score_" + dataset + "_" + mode + "_dur.pkl")
            if os.path.isfile(dur_filepath):
                dur_r2 = '{:.4f}'.format(round(joblib.load(dur_filepath),4))
            else:
                dur_r2 = ""
            table_rows[0].append(rew_r2)
            table_rows[1].append(dur_r2)
    # Output to csv
    output_filepath = os.path.join(estimates_path, "r2table.csv")
    with open(output_filepath, "w") as f:
        writer = csv.writer(f)
        writer.writerows(table_rows)

def main():
    generateR2Table()

if __name__ == "__main__":
    main()