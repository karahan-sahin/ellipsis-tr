
import numpy as np
from datasets import DatasetDict

def annotateSpanIds(examples, label_list, label2id):

    tokenized_data = {
        'id': [],
        'text': [],
        'tokens': [],
        'ner_tags': [],
    }

    for _id, (text, spans, label) in enumerate(zip(examples['text'],examples['span'], examples['label'])):
        tokenized_text = text.split(' ')
        ids = np.zeros(len(tokenized_text), dtype=int)
        for span in spans:
            span_words = span.split(' ')
            len_key = len(span_words)
            for token_idx, token in enumerate(tokenized_text):
                if token == span_words[0]:
                    ids[token_idx] = label2id['B-ANTECEDENT' if len(label_list) == 3 else f'B-{label}']
                    if len_key:ids[token_idx+1: token_idx+len_key] = label2id['I-ANTECEDENT' if len(label_list) == 3 else f'I-{label}']

        tokenized_data['text'].append(text)
        tokenized_data['id'].append(_id)
        tokenized_data['tokens'].append(tokenized_text)
        tokenized_data['ner_tags'].append(ids)

    return DatasetDict(tokenized_data)


def tokenize_and_align_labels(examples, tokenizer):
        tokenized_inputs = tokenizer(examples["tokens"], padding=True, truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs