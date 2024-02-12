
import os
import evaluate
import numpy as np
import pandas as pd
from datasets import DatasetDict
from lib.dataset.dataset import Enviroment

seqeval = evaluate.load("seqeval")
accuracy = evaluate.load("accuracy")
precision = evaluate.load('precision')
recall = evaluate.load('recall')
f1 = evaluate.load('f1')
rouge = evaluate.load("rouge")

def compute_metrics_for_span_annotation(
    label_list: list[str], label_str: list[str]
    ) -> function:
    """This wrapper function takes a list of labels and a list of label strings and returns a function that computes metrics for span annotation.

    Args:
        label_list list[str]: _description_
        label_str (_type_): _description_
        
    Returns:
        compute_metrics (function): 
    """

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        RESULTS = {
            "Overall Precision": results["overall_precision"],
            "Overall Recall": results["overall_recall"],
            "Overall F1": results["overall_f1"],
            "Overall Accuracy": results["overall_accuracy"],
        }

        for label in label_str:
            RESULTS.update( {f'{label} {metric.capitalize()}': results[label][metric] 
                             for metric in ['precision', 'recall', 'f1'] 
                                if label in results.keys() } )

        dfff = pd.DataFrame({label: [score] for label, score in RESULTS.items()})

        CURRENT_EPOCH = 0
        file_path = f"{Enviroment.OUTPUT_LOG_DIR}/ellipsis-tr-{Enviroment.MODEL}-{Enviroment.TASK_TYPE}-{Enviroment.SPAN_TYPE}-{Enviroment.EXP_TYPE}.xlsx"

        if not os.path.exists(file_path): dfff.to_excel(file_path, sheet_name='Epoch 0')
        else:
            test_ = pd.ExcelFile(file_path)
            CURRENT_EPOCH = len(test_.sheet_names)
            with pd.ExcelWriter(file_path, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer:
                dfff.to_excel(writer, sheet_name=f'Epoch {CURRENT_EPOCH}')

        return RESULTS

    return compute_metrics


def compute_metrics_for_type_classification():

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy.compute(predictions=predictions, references=labels)['accuracy'],
            'precision': precision.compute(predictions=predictions, references=labels, average='macro')['precision'],
            'recall': recall.compute(predictions=predictions, references=labels, average='macro')['recall'],
            'f1': f1.compute(predictions=predictions, references=labels, average='macro')['f1']
        }

    return compute_metrics


def compute_metrics_for_correlate_generation(tokenizer):

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics


def preprocess_wrapper_for_correlate(tokenizer):
    
    def preprocess_function(examples):
        inputs = [doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(text_target=examples["correlate"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function