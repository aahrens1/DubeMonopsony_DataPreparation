# Train title, description, and keyword doc2vec models
import argparse
import gensim
import joblib
import multiprocessing
import nltk
import os
import pandas as pd

# The dimension of the resulting vectors
VEC_DIM = 50

# Inputs required:
#
# Models:
# panos_titles_doc2vec.pkl, panos_desc_doc2vec.pkl, panos_kw_doc2vec.pkl
#
# Document tags:
# panos_titles_doc2vec.pkl.docvecs.doctag_syn0.npy, ...
#
# Corpora:
# panos_titles_doc2vec_corpus.pkl, panos_desc_doc2vec_corpus.pkl, panos_kw_doc2vec_corpus.pkl

cleaned_path = os.path.join("..","cleaned_data")
doc2vec_path = os.path.join("..","doc2vec_output")

def loadModel(pickle_filename):
    # Loads the .pkl file into a gensim doc2vec object
    return gensim.models.doc2vec.Doc2Vec.load(os.path.join(doc2vec_path,pickle_filename))

def generateCorpora(dataset):
    if dataset == "ipeirotis":
        ## Load Ipeirotis data
        full_df = pd.read_pickle(os.path.join(cleaned_path,"ipeirotis_cleaned.pkl"))
        full_df["title"] = full_df["title"].astype(str)
        full_df["description"] = full_df["description"].astype(str)
    elif dataset == "textlab_30":
        ## Load TextLab 30-minute data
        full_df = pd.read_pickle(os.path.join(cleaned_path,"textlab_30_cleaned.pkl"))
        full_df["title"] = full_df["title"].astype(str)
    elif dataset == "textlab_10":
        ## Load TextLab 10-minute data
        full_df = pd.read_pickle(os.path.join(cleaned_path,"textlab_10_cleaned.pkl"))
        full_df["description"] = full_df["description"].astype(str)

    # Create (empty) models
    title_model = gensim.models.doc2vec.Doc2Vec(vector_size=VEC_DIM,min_count=2)
    if dataset != "textlab_30":
        desc_model = gensim.models.doc2vec.Doc2Vec(vector_size=VEC_DIM,min_count=2)
        kw_model = gensim.models.doc2vec.Doc2Vec(vector_size=VEC_DIM,min_count=2)

    # Extract docs
    titles = full_df["title"]
    if dataset != "textlab_30":
        descriptions = full_df["description"]
        keywords = full_df["kw_parsed"]

    # Load stopwords
    stoplist = nltk.corpus.stopwords.words("english")

    # Preprocess and generate the corpora
    def preprocess(text):
        word_list = gensim.utils.simple_preprocess(text)
        word_list = [word for word in word_list if (word not in stoplist) and (len(word) > 2)]
        return word_list

    def generateCorpus(doc_series):
        corpus = [gensim.models.doc2vec.TaggedDocument(preprocess(doc), [doc_id])
              for doc_id,doc in doc_series.iteritems()]
        return corpus

    title_corpus = generateCorpus(titles)
    if dataset != "textlab_30":
        desc_corpus = generateCorpus(descriptions)
        kw_corpus = generateCorpus(keywords)
    joblib.dump(title_corpus, os.path.join(doc2vec_path,dataset + "_titles_doc2vec_corpus.pkl"))
    if dataset != "textlab_30":
        joblib.dump(desc_corpus, os.path.join(doc2vec_path,dataset + "_desc_doc2vec_corpus.pkl"))
        joblib.dump(kw_corpus, os.path.join(doc2vec_path,dataset + "_kw_doc2vec_corpus.pkl"))
    print("Corpora exported")

    # Build vocab
    title_model.build_vocab(title_corpus)
    if dataset != "textlab_30":
        desc_model.build_vocab(desc_corpus)
        kw_model.build_vocab(kw_corpus)

    # Save model to send to AWS
    title_model.save(os.path.join(doc2vec_path,dataset + "_titles_doc2vec.pkl"))
    if dataset != "textlab_30":
        desc_model.save(os.path.join(doc2vec_path,dataset + "_desc_doc2vec.pkl"))
        kw_model.save(os.path.join(doc2vec_path,dataset + "_kw_doc2vec.pkl"))
    print("Models saved")

def trainModels(dataset):
    # Loads the doc2vec models generated in generateCorpora() and performs the training
    num_cpus = multiprocessing.cpu_count()

    # Load model .pkls
    title_model = loadModel(os.path.join(doc2vec_path,dataset + "_titles_doc2vec.pkl"))
    if dataset != "textlab_30":
        desc_model = loadModel(os.path.join(doc2vec_path,dataset + "_desc_doc2vec.pkl"))
        kw_model = loadModel(os.path.join(doc2vec_path,dataset + "_kw_doc2vec.pkl"))

    # Load corpus .pkls
    title_corpus = joblib.load(os.path.join(doc2vec_path,dataset + "_titles_doc2vec_corpus.pkl"))
    if dataset != "textlab_30":
        desc_corpus = joblib.load(os.path.join(doc2vec_path,dataset + "_desc_doc2vec_corpus.pkl"))
        kw_corpus = joblib.load(os.path.join(doc2vec_path,dataset + "_kw_doc2vec_corpus.pkl"))

    # Train model
    print("Training title model")
    title_model.train(title_corpus, total_examples=title_model.corpus_count,
        epochs=title_model.iter)
    title_model.save(os.path.join(doc2vec_path,dataset + "_titles_doc2vec_trained.pkl"))
    if dataset != "textlab_30":
        print("Training description model")
        desc_model.train(desc_corpus, total_examples=desc_model.corpus_count,
            epochs=desc_model.iter)
        desc_model.save(os.path.join(doc2vec_path,dataset + "_desc_doc2vec_trained.pkl"))
        print("Training keyword model")
        kw_model.train(kw_corpus, total_examples=kw_model.corpus_count,
            epochs=kw_model.iter)
        kw_model.save(os.path.join(doc2vec_path,dataset + "_kw_doc2vec_trained.pkl"))
    print("Trained models saved")

def inferVectors(dataset):
    # Load the trained models
    titles_trained = loadModel(os.path.join(doc2vec_path,dataset + "_titles_doc2vec_trained.pkl"))
    if dataset != "textlab_30":
        desc_trained = loadModel(os.path.join(doc2vec_path,dataset + "_desc_doc2vec_trained.pkl"))
        kw_trained = loadModel(os.path.join(dataset + "_kw_doc2vec_trained.pkl"))

    # Load corpora
    title_corpus = joblib.load(os.path.join(doc2vec_path,dataset + "_titles_doc2vec_corpus.pkl"))
    if dataset != "textlab_30":
        desc_corpus = joblib.load(os.path.join(doc2vec_path,dataset + "_desc_doc2vec_corpus.pkl"))
        kw_corpus = joblib.load(os.path.join(doc2vec_path,dataset + "_kw_doc2vec_corpus.pkl"))

    # Get vectors for all of the documents (descriptions) in the corpus
    print("Inferring title vectors")
    title_corpus_vecs = [titles_trained.infer_vector(corpus_doc.words) for corpus_doc in title_corpus]
    if dataset != "textlab_30":
        print("Inferring description vectors")
        desc_corpus_vecs = [desc_trained.infer_vector(corpus_doc.words) for corpus_doc in desc_corpus]
        print("Inferring keyword vectors")
        kw_corpus_vecs = [kw_trained.infer_vector(corpus_doc.words) for corpus_doc in kw_corpus]
    title_col_headers = ["doc2vec_title_" + str(dim) for dim in range(VEC_DIM)]
    if dataset != "textlab_30":
        desc_col_headers = ["doc2vec_desc_" + str(dim) for dim in range(VEC_DIM)]
        kw_col_headers = ["doc2vec_kw_" + str(dim) for dim in range(VEC_DIM)]
    title_vec_df = pd.DataFrame(title_corpus_vecs, columns=title_col_headers)
    if dataset != "textlab_30":
        desc_vec_df = pd.DataFrame(desc_corpus_vecs, columns=desc_col_headers)
        kw_vec_df = pd.DataFrame(kw_corpus_vecs, columns=kw_col_headers)
    if dataset == "textlab_30":
        df = title_vec_df
    else:
        df = pd.concat([title_vec_df, desc_vec_df, kw_vec_df], axis=1)
    df.to_pickle(os.path.join(doc2vec_path,dataset + "_doc2vec.pkl"))

def doc2VecPipeline(dataset):
    # Before anything else, make sure there's a doc2vec output folder
    if not os.path.isdir(doc2vec_path):
        os.mkdir(doc2vec_path)
    generateCorpora(dataset)
    trainModels(dataset)
    inferVectors(dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    doc2VecPipeline(args.dataset)
    

if __name__ == '__main__':
    main()