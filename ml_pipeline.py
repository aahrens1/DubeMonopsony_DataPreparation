# Basically just a wrapper file that runs all the individual parts of the ML
# pipeline in a loop, for the provided dataset ("textlab_10", textlab_30", or
# "ipeirotis")
import argparse
import joblib
import os
import sys

from global_settings import output_paths

from compute_group_stats import computeGroupStats
from clean_group_data import cleanGroupData
from mturk_lda import LDAPipeline
from mturk_doc2vec import doc2VecPipeline
from prepare_ml import prepareML
from run_ml import runML
from most_predictive_features import computeMostPredictive
from double_ml_estimate import computeEta
from gen_r2_table import generateR2Table

all_datasets = ["ipeirotis", "textlab_30", "textlab_10"]

def runPipeline(dataset):
    # First things first, make sure the output directories exist. If not, create them
    for path_name in output_paths:
        output_path = output_paths[path_name]
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

    # First step: take the raw _panel.csv and _meta.csv datasets and combine
    # them into a HIT-group-level _group.csv dataset
    print("*** Computing group-level statistics")
    computeGroupStats(dataset)

    # This just cleans the data (Winsorizes) and makes sure everything is in the
    # correct data format for the rest of the pipeline
    print("*** Cleaning data")
    cleanGroupData(dataset)

    # Now pre-compute the doc2vec and LDA matrices, for use in the rest of the
    # pipeline
    print("*** Precomputing doc2vec vectors")
    doc2VecPipeline(dataset)
    print("*** Precomputing LDA vectors")
    LDAPipeline(dataset)

    # Next, compute the features based on the group-level dataset and split into
    # training and test data, then run the actual ML (for A->B and B->A modes)
    
    # A->B run with just n-gram features
    print("*** A->B n-gram ML run")
    prepareML(dataset, "gramab")
    runML(dataset, "gramab")
    # Compute and save the most predictive features
    print("*** A->B computing most predictive features")
    computeMostPredictive(dataset, "gramab")
    # Now the second ML run, using the most predictive features plus the rest
    # of the features detailed in Appendix D
    print("*** A->B full ML run")
    prepareML(dataset, "fullab")
    runML(dataset, "fullab")

    # Repeat the above, but for the B->A run
    print("*** B->A n-gram ML run")
    prepareML(dataset, "gramba")
    runML(dataset, "gramba")
    print("*** B->A computing most predictive features")
    computeMostPredictive(dataset, "gramba")
    print("*** B->A full ML run")
    prepareML(dataset, "fullba")
    runML(dataset, "fullba")

    # Finally, compute the double ML estimate using the ML residuals
    print("*** Computing final point estimate, exporting Stata residual file")
    computeEta(dataset)
    

def main():
    # Parse command-line args
    if len(sys.argv) > 1:
        # Specific dataset passed in, just run the pipeline on the dataset
        dataset = sys.argv[1]
        runPipeline(dataset)
    else:
        # No arguments given, so loop over all datasets
        for cur_dataset in all_datasets:
            print("***** Running pipeline for dataset " + cur_dataset)
            runPipeline(cur_dataset)

    # Regardless of these choices, generate the R^2 table
    generateR2Table()
    

if __name__ == "__main__":
    main()