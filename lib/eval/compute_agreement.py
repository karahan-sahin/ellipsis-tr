import pandas as pd
from sklearn.metrics import cohen_kappa_score
import argparse

def compute_agreement(datafile):
    # Load the dataset
    df = pd.read_json(datafile, lines=True)
    
    # Ensure the dataframe has the required columns
    required_columns = ['sentence_id', 'sentence', 'annotation_label', 'annotator_task', 'annotator_id']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"Datafile must contain the following columns: {required_columns}")
    
    # Pivot the dataframe to have annotators as columns and sentences as rows
    pivot_df = df.pivot_table(index='sentence_id', columns='annotator_id', values='annotation_label', aggfunc='first')
    
    # Drop rows with any missing values
    pivot_df = pivot_df.dropna()
    
    # Compute pairwise Cohen's Kappa for all annotator pairs
    annotators = pivot_df.columns
    kappa_scores = {}
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            annotator1 = annotators[i]
            annotator2 = annotators[j]
            kappa = cohen_kappa_score(pivot_df[annotator1], pivot_df[annotator2])
            kappa_scores[(annotator1, annotator2)] = kappa
    
    return kappa_scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute interannotator agreement.")
    parser.add_argument("datafile", type=str, help="Path to the dataset file.")
    args = parser.parse_args()

    kappa_scores = compute_agreement(args.datafile)
    for annotator_pair, kappa in kappa_scores.items():
        print(f"Cohen's Kappa between {annotator_pair[0]} and {annotator_pair[1]}: {kappa:.2f}")