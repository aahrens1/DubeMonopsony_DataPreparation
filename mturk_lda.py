# Prepares and runs LDA on the titles, descriptions, and keywords
import argparse
import logging
import os
import string

import pandas as pd
from nltk.corpus import stopwords
import gensim

from global_settings import output_paths

remove_punct = str.maketrans({key: None for key in string.punctuation})

cleaned_path = output_paths["cleaned"]
lda_path = output_paths["lda"]

# Number of cores to use (to avoid filling up our cluster I use 7, it doesn't
# take all that long so no real need to use all 32)
num_workers = 7

def generateCorpora(dataset):
    if dataset == "ipeirotis":
        ## Load the Ipeirotis dataset
        #full_df = pd.read_pickle("merged_durations.pkl")
        full_df = pd.read_pickle(os.path.join(cleaned_path,"ipeirotis_cleaned.pkl"))
        full_df["title"] = full_df["title"].astype(str)
        full_df["description"] = full_df["description"].astype(str)
    elif dataset == "textlab_30":
        ## Load the TextLab 30-minute dataset
        full_df = pd.read_pickle(os.path.join(cleaned_path,"textlab_30_cleaned.pkl"))
        full_df["title"] = full_df["title"].astype(str)
    elif dataset == "textlab_10":
        ## Load the TextLab 10-minute dataset
        full_df = pd.read_pickle(os.path.join(cleaned_path,"textlab_10_cleaned.pkl"))
        # Make sure the descriptions are all strings (-__-)
        full_df["description"] = full_df["description"].astype(str)

    stoplist = stopwords.words('english')
    stoplist.extend(["<p>","</p>","<p></p>","</p><p>","<i>","</i>","\t"])

    # Extract docs
    title_df = full_df["title"]
    if dataset != "textlab_30":
        desc_df = full_df["description"]
        kw_df = full_df["kw_parsed"]
    # Remove punctuation
    title_df = title_df.str.translate(remove_punct)
    if dataset != "textlab_30":
        desc_df = desc_df.str.translate(remove_punct)
        kw_df = kw_df.str.translate(remove_punct)

    # Create doc lists
    def seriesToDocList(doc_series):
        docs = [[cur_word for cur_word in cur_doc.lower().split() if cur_word not in stoplist] for cur_doc in doc_series]
        docs = [[cur_word for cur_word in cur_doc if len(cur_word) > 2] for cur_doc in docs]
        return docs
    title_docs = seriesToDocList(title_df)
    if dataset != "textlab_30":
        desc_docs = seriesToDocList(desc_df)
        kw_docs = seriesToDocList(kw_df)
    print("Doc lists created")

    # Create gensim dictionaries
    title_dict = gensim.corpora.Dictionary(title_docs)
    if dataset != "textlab_30":
        desc_dict = gensim.corpora.Dictionary(desc_docs)
        kw_dict = gensim.corpora.Dictionary(kw_docs)

    # Save to .dict files
    title_dict.save(os.path.join(lda_path, dataset + "_title.dict"))
    if dataset != "textlab_30":
        desc_dict.save(os.path.join(lda_path, dataset + "_desc.dict"))
        kw_dict.save(os.path.join(lda_path, dataset + "_kw.dict"))
    print("Dictionaries saved")

    # Create gensim corpora
    title_corpus = [title_dict.doc2bow(doc) for doc in title_docs]
    if dataset != "textlab_30":
        desc_corpus = [desc_dict.doc2bow(doc) for doc in desc_docs]
        kw_corpus = [kw_dict.doc2bow(doc) for doc in kw_docs]

    # Serialize corpora
    title_corpus_path = os.path.join(lda_path, dataset + "_title_corpus.mm")
    gensim.corpora.MmCorpus.serialize(title_corpus_path, title_corpus)
    if dataset != "textlab_30":
        desc_corpus_path = os.path.join(lda_path, dataset + "_desc_corpus.mm")
        gensim.corpora.MmCorpus.serialize(desc_corpus_path, desc_corpus)
        kw_corpus_path = os.path.join(lda_path, dataset + "_kw_corpus.mm")
        gensim.corpora.MmCorpus.serialize(kw_corpus_path, kw_corpus)
    print("Corpora serialized")

def computeLDA(K, data_prefix):
    doc_dict = gensim.corpora.dictionary.Dictionary.load(os.path.join(lda_path, data_prefix + ".dict"))
    doc_corpus = gensim.corpora.mmcorpus.MmCorpus(os.path.join(lda_path, data_prefix + "_corpus.mm"))
    lda = gensim.models.ldamulticore.LdaMulticore(corpus=doc_corpus,
      id2word=doc_dict, num_topics=K, workers=7)
    lda_filename = os.path.join(lda_path, data_prefix + "_lda_" + str(K) + ".pkl")
    lda.save(lda_filename)
  
def getTopicDistributions(K, data_prefix):
    # Now get the topic distribution for each HIT batch and save
    #full_df = pd.read_pickle("merged_durations.pkl")
    prefix_var = data_prefix.split("_")[-1]
    lda_filename = os.path.join(lda_path, data_prefix + "_lda_" + str(K) + ".pkl")
    lda = gensim.models.ldamulticore.LdaMulticore.load(lda_filename)
    corpus_filename = os.path.join(lda_path, data_prefix + "_corpus.mm")
    corpus = gensim.corpora.mmcorpus.MmCorpus(corpus_filename)
    # You can pass in a collection to the lda object, so the result of this
    # line will be the topic distributions for all 5mil descriptions
    corpus_dist = lda[corpus]
    print("Distributions computed")
    corpus_dict = [{elt[0]:elt[1] for elt in doc_dist} for doc_dist in corpus_dist]
    print("Dict created")
    corpus_arr = [[doc_dict[key] if (key in doc_dict) else 0.0 for key in range(K)] for doc_dict in corpus_dict]
    print("Array created")
    column_names = [prefix_var + "topic_" + str(K) + "_" + str(n) for n in range(K)]
    corpus_df = pd.DataFrame(corpus_arr,columns=column_names)
    print("DF created")
    dist_file = os.path.join(lda_path, data_prefix + "_lda_dists_" + str(K) + ".pkl")
    corpus_df.to_pickle(dist_file)
    print(corpus_df.head())

def LDAPipeline(dataset):
    # Before anything else, make sure there's an lda output folder
    if not os.path.isdir(lda_path):
        os.mkdir(lda_path)

    if dataset == "textlab_10":
        data_prefixes = ["textlab_10_title","textlab_10_desc","textlab_10_kw"]
    elif dataset == "textlab_30":
        data_prefixes = ["textlab_30_title"]
    elif dataset == "ipeirotis":
        data_prefixes = ["ipeirotis_title","ipeirotis_desc","ipeirotis_kw"]

    # Generate the corpora (currently done in Jupyter, should be done here in the future)
    generateCorpora(dataset)

    K_vals = [5,10,15,20]
    for K in K_vals:
        for data_prefix in data_prefixes:
            computeLDA(K, data_prefix)
            getTopicDistributions(K, data_prefix)

def main():
    # Parse the command-line args before running the main pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    LDAPipeline(args.dataset)

if __name__ == "__main__":
    main()