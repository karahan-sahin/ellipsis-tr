"""This module contains the functions to augment the dataset. 
The functions are used to extract the lemma list, parse the dependency annotations, and augment the dataset.

"""

import os
import re
import nltk
import time
import random
import stanza
import requests
import numpy as np
import pandas as pd
from dataset import Annotation
from deep_translator import GoogleTranslator

from tqdm.notebook import tqdm
tqdm.pandas()


def get_lemma_list() -> dict:
    """This function extracts the lemma list from the zemberek-python repository and creates a dictionary with the lemma as the key and the POS tag as the value.
    
    Returns:
        dict: A dictionary with the lemma as the key and the POS tag as the value.
    """


    file = requests.get("https://raw.githubusercontent.com/Loodos/zemberek-python/master/zemberek/resources/lexicon.csv")

    # You only need to run this code as well
    word_to_pos_dictionary = {line.split("\t")[2]: (line.split("\t")[1][0].isupper(), line.split("\t")[3], line.split("\t")[4])
                                for line in file.text.split("\n")[:-1]
                                    if line.split("\t")[2].isalpha()
                                and len(line.split("\t")[2]) > 2 }

    print("Lemma list is extracted")
    
    for k, v in word_to_pos_dictionary.items(): word_to_pos_dictionary[k] = "_".join([x for x in v if isinstance(x,str) and x != "None"])
    
    return word_to_pos_dictionary


def get_augment_items(word_to_pos_dictionary: dict) -> dict:
    
    PROPN = [re.sub("i̇","i",word.lower().capitalize()) for word in open("isimler.txt", "r").read().split("\n")]
    NUMS = [word for word, pos in word_to_pos_dictionary.items() if "Num" in pos]
    ADJS = [word for word, pos in word_to_pos_dictionary.items() if "Adj" == pos]
    VERBS = [word for word, pos in word_to_pos_dictionary.items() if "Verb" == pos]
    NOUNS = [word for word, pos in word_to_pos_dictionary.items() if "Noun" == pos]
    
    return PROPN, NUMS, ADJS, VERBS, NOUNS
    
def parse_data_dependency(data: pd.DataFrame) -> pd.DataFrame:
    """ This function uses the stanza library to extracts the depencency annotations data and add the parses to the dataset.
    
    Parameters:
        data (pd.DataFrame): The dataset to be processed.
    
    Returns:
        pd.DataFrame: The processed dataset with the parses added.
    """
    
    nlp = stanza.Pipeline(
        lang='tr', 
        processors="""tokenize, mwt, pos, lemma, depparse""", 
        use_gpu=True
    )
    
    def get_parse(sentence): return nlp(sentence).sentences[0].to_dict()

    data["parses"] = data.candidate_text.progress_apply(get_parse)

    return data


def augmentType(record: Annotation,
                augment_count: int = 1, 
                PROP: bool = False, 
                NUM: bool = False, 
                NOUN: bool = False, 
                VERB: bool = False, 
                ADJ: bool = False) -> list:
    """This function takes a record and returns a list of augmented records. 
    It takes the record and the POS tags to augment as input.
    
    Args:
        record (Annotation): The record to be augmented.
        PROP (bool, optional): Whether to augment the proper nouns. Defaults to False.
        NUM (bool, optional): Whether to augment the numbers. Defaults to False.
        NOUN (bool, optional): Whether to augment the nouns. Defaults to False.
        VERB (bool, optional): Whether to augment the verbs. Defaults to False.
        ADJ (bool, optional): Whether to augment the adjectives. Defaults to False.

    Returns:
        list: A list of augmented records.
    """

    tokens = record['parses']

    augments = []

    # If the span is not a list, make it a list
    if not isinstance(record['span'], list): record['span'] = []

    for _ in range(augment_count):
        tokens_ = []
        x = False
        for idx, word in enumerate(tokens):
            
            # Check if the word is split token
            if not isinstance(word["id"], tuple):
                if (word["xpos"] == "PROPN" or word["upos"] == "PROPN") and PROP:
                    x = True
                    aug = random.choice(PROPN)
                    tokens_.append({"text": aug})
                    for idx, span in enumerate(record['span']):
                        if word['text'] == span:
                            record['span'][idx] = aug
                        elif word['text'] in span:
                            record['span'][idx].replace(word['text'], aug)

                elif (word["upos"] == "NUM") and NUM:
                    x = True
                    aug = random.choice(NUMS)
                    tokens_.append({"text": aug})
                    for idx, span in enumerate(record['span']):
                        if word['text'] == span: record['span'][idx] = aug
                        elif word['text'] in span: record['span'][idx].replace(word['text'], aug)
                
                elif (word["upos"] == "NOUN") and NOUN:
                    x = True
                    aug = random.choice(NOUNS)
                    tokens_.append({"text": aug})
                    for idx, span in enumerate(record['span']):
                        if word['text'] == span: record['span'][idx] = aug
                        elif word['text'] in span: record['span'][idx].replace(word['text'], aug)
                        
                
                # For Noun and Verb Pls Check the morphological features as well!!
                elif (word["upos"] == "VERB") and VERB:
                    x = True
                    aug = random.choice(VERBS)
                    tokens_.append({"text": aug})
                    for idx, span in enumerate(record['span']):
                        if word['text'] == span: record['span'][idx] = aug
                        elif word['text'] in span: record['span'][idx].replace(word['text'], aug)
                        
                        
                elif (word["upos"] == "ADJ") and ADJ:
                    x = True
                    aug = random.choice(ADJS)
                    tokens_.append({"text": aug})
                    for idx, span in enumerate(record['span']):
                        if word['text'] == span: record['span'][idx] = aug
                        elif word['text'] in span: record['span'][idx].replace(word['text'], aug)

                else: tokens_.append(word)


    # If the record is augmented, add it to the list
    if x: augments.append({
        'candidate_text': " ".join([i["text"] for i in tokens_]),
        'elliptical_type': record['elliptical_type'],
        'sentence_quality': record['sentence_quality'],
        'span': record['span'],
      })

    return augments


def add_augmented_data(data: pd.DataFrame, 
                       PROP: bool = False,
                       NUM: bool = False,
                       NOUN: bool = False,
                       VERB: bool = False,
                       ADJ: bool = False, 
                       augment_per_type: int = 1,
                       export_augments: bool = True) -> pd.DataFrame:
    """This function takes a dataset and returns a dataset with augmented records.
    
    Args:
        data (pd.DataFrame): The dataset to be augmented.
        PROP (bool, optional): Whether to augment the proper nouns. Defaults to False.
        NUM (bool, optional): Whether to augment the numbers. Defaults to False.
        NOUN (bool, optional): Whether to augment the nouns. Defaults to False.
        VERB (bool, optional): Whether to augment the verbs. Defaults to False.
        ADJ (bool, optional): Whether to augment the adjectives. Defaults to False.
        augment_per_type (int, optional): The number of augmented records to be added for each type. Defaults to 1.        
        export_augments (bool, optional): Whether to export the augmented records. Defaults to True.

    Returns:
        pd.DataFrame: The dataset with augmented records.
    """
    
    # For each record, augment for each type separately,
    # then add the augmented records to the dataset
    AUGMENTS = []
    for idx, record in tqdm(data.iterrows(), total=len(data)):
        if PROP: AUGMENTS += augmentType(record, augment_count=augment_per_type, PROP=True)
        if NUM: AUGMENTS += augmentType(record, augment_count=augment_per_type, NUM=True)
        if NOUN: AUGMENTS += augmentType(record, augment_count=augment_per_type, NOUN=True)
        if VERB: AUGMENTS += augmentType(record, augment_count=augment_per_type, VERB=True)
        if ADJ: AUGMENTS += augmentType(record, augment_count=augment_per_type, ADJ=True)
    
    
    if export_augments: 
        CURRENT_TIME = time.strftime("%Y%m%d-%H%M%S")
        EXPORT_PATH = os.path.join(os.environ['AUGMENT_PATH'] , f"augments_{len(AUGMENTS)}_{PROP=}_{NUM=}_{NOUN=}_{VERB=}_{ADJ=}_{CURRENT_TIME}.xlsx")
        pd.DataFrame(AUGMENTS).to_excel(EXPORT_PATH, index=False)
    
    return pd.concat([data, pd.DataFrame(AUGMENTS)])




def augment_with_backtranslation():
    """This function takes the dataset and augments it with backtranslation.
    
    
    """

    to_en = GoogleTranslator(source='tr', target='en')
    to_tr = GoogleTranslator(source='en', target='tr')

    sim_ann0 = pd.read_csv("data/top_similarity_test_case_6_10 - top_similarity_test_case_6.csv")
    sim_ann1 = pd.read_csv("data/top_similarity_test_case_16_20 - top_similarity_test_case_16.csv.csv")
    sim_ann2 = pd.read_csv("data/top_similarity_test_case_21_25 - top_similarity_test_case_21.csv.csv")

    PROPN = pd.read_csv("data/isimler.txt")
    arg_drop = pd.read_csv("data/Seed Data - Argument Drop.csv")
    np_drop = pd.read_csv("data/Seed Data - NP Drop.csv")
    obj_drop = pd.read_csv("data/Seed Data - Object Drop.csv")
    annotations = pd.read_csv("data/data_final.csv")
    
    
    annotations = annotations[annotations.ellipsis_type_1 != "Subject Drop"][annotations.ellipsis_type_1.apply(lambda x: True if not pd.isna(x) else False)]
    similarity = pd.concat([sim_ann0,sim_ann1,sim_ann2])
    similarity = similarity[similarity.annotation == "1"]

    arg_drop.columns = ['text']
    np_drop.columns = ['text']
    obj_drop.columns = ['text']
    similarity.columns = ['text', 'annotation', 'ellipsis_type_1', 'ellipsis_type_2', 'parse', 'similarity']

    def multi_iter_translation(sentence, fout, source):
        fw_0 = to_en.translate(sentence)
        bw_0 = to_tr.translate(fw_0)

        if bw_0 != sentence:
            fout.write(f"{sentence},{bw_0},v0,{source}\n")
            fw_1 = to_en.translate(bw_0)
            bw_1 = to_tr.translate(fw_1)

            if bw_0 != sentence and bw_0 != bw_1:
                fout.write(f"{sentence},{bw_1},v1,{source}\n")
                fw_2 = to_en.translate(bw_1)
                bw_2 = to_tr.translate(fw_2)

                if bw_0 != sentence and bw_0 != bw_1 and bw_1 != bw_2 and bw_0 != bw_2:
                    fout.write(f"{sentence},{bw_2},v2,{source}\n")

    with open("translation_aug_2.csv", "w+", encoding="utf-8") as fout:
        #for idx, sentence in enumerate(arg_drop.text.to_list()):
        #    print(f"Running at idx: {idx} on arg_drop")
        #    multi_iter_translation(sentence, fout, "arg_drop")

        #for idx, sentence in enumerate(np_drop.text.to_list()):
        #    print(f"Running at idx: {idx} on np_drop")
        #    multi_iter_translation(sentence, fout, "np_drop")

        #for idx, sentence in enumerate(obj_drop.text.to_list()):
        #    print(f"Running at idx: {idx} on obj_drop")
        #    multi_iter_translation(sentence, fout, "obj_drop")

        for idx, sentence in list(enumerate(similarity.text.to_list()))[1254:]:
            print(f"Running at idx: {idx} on similarity")
            multi_iter_translation(sentence, fout, "similarity")
            
        for idx, sentence in enumerate(annotations.text.to_list()):
            print(f"Running at idx: {idx} on annotations")
            multi_iter_translation(sentence, fout, "annotations")