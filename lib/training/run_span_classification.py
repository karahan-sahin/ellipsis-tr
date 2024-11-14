import os
import torch
import wandb
import evaluate
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments

from dotenv import load_dotenv
load_dotenv()

def init_wandb(run_name):
    
    if not args.use_wandb:
        os.environ['WANDB_DISABLED'] = 'true'
        return

    wandb.login(
        key=os.getenv('WANDB_API_KEY')
    )
    wandb.init(
        entity="boun-pilab",
        project="ellipsis-tr",
        name=run_name,
    )

def parse_args():

    import argparse
    parser = argparse.ArgumentParser(description='Train a span classification model')

    # Required arguments
    parser.add_argument('--dataset_file', type=str, required=True, help='Path to the CSV file containing the data')
    
    parser.add_argument('--model_name', type=str, default='bert-base-cased', help='Model name')
    parser.add_argument('--extraction_type' , type=str, default='discriminative', help='Extraction type')
    parser.add_argument('--hub_model_id', type=str, default=None, help='Hub model ID')

    parser.add_argument('--num_labels', type=int, default=9, help='Number of labels')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--logging_steps', type=int, default=1, help='Logging steps')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16, help='Per device train batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16, help='Per device eval batch size')
    parser.add_argument('--save_steps', type=int, default=500, help='Save steps')
    parser.add_argument('--eval_steps', type=int, default=500, help='Eval steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--report_to', type=str, default='wandb', help='Report to')
    parser.add_argument('--push_to_hub', action='store_true', help='Push to hub')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # Create a new run name
    run_name = f"{args.model_name.split('/')[0]}-{args.extraction_type}-span-classification"
    
    init_wandb(run_name)

    # Load dataset from csv file
    train_df = pd.read_csv(args.dataset_file)
    val_df = pd.read_csv(args.dataset_file.replace('train', 'val'))
    test_df = pd.read_csv(args.dataset_file.replace('train', 'test'))

    if args.extraction_type == 'discriminative':
        label_list = [ 'O', 'B-ANTECEDENT', 'I-ANTECEDENT']

    elif args.extraction_type == 'extractive':
        label_list = [ "O" ] + [ f"B-{label}" for label in train_df.elliptical_type.unique() ] + [ f"I-{label}" for label in train_df.elliptical_type.unique() ]

    label2id = {label: i for i, label in enumerate(label_list)}

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list))

    # Read tokenized_text and discriminative_span and extractive_span columns as lists,
    def read_literal(df):
        from ast import literal_eval
        df['tokenized_text'] = df['tokenized_text'].apply(literal_eval)
        df['tokenized_span'] = df['tokenized_span'].apply(literal_eval)
        df['discriminative_span'] = df['discriminative_span'].apply(literal_eval)
        df['extractive_span'] = df['extractive_span'].apply(literal_eval)
        return df
    
    train_df = read_literal(train_df)
    val_df = read_literal(val_df)
    test_df = read_literal(test_df)

    train_df['tokenized_text'] = train_df['candidate_text'].apply(tokenizer.tokenize)
    val_df['tokenized_text'] = val_df['candidate_text'].apply(tokenizer.tokenize)
    test_df['tokenized_text'] = test_df['candidate_text'].apply(tokenizer.tokenize)

    train_df['tokenized_span'] = train_df['span'].apply(tokenizer.tokenize)
    val_df['tokenized_span'] = val_df['span'].apply(tokenizer.tokenize)
    test_df['tokenized_span'] = test_df['span'].apply(tokenizer.tokenize)

    from string import punctuation
    # Remove if punctuations are at the beginning or end of the span
    def remove_punctuations(row):
        if row['tokenized_span'][0] in punctuation:
            row['tokenized_span'] = row['tokenized_span'][1:]
        if row['tokenized_span'][-1] in punctuation:
            row['tokenized_span'] = row['tokenized_span'][:-1]
        return row
    
    train_df = train_df.apply(remove_punctuations, axis=1)
    val_df = val_df.apply(remove_punctuations, axis=1)
    test_df = test_df.apply(remove_punctuations, axis=1)

    # Create ner_tags column
    def create_ner_tags(row):
        ner_tags = ['O'] * len(row['tokenized_text'])
        for idx, token in enumerate(row['tokenized_text']):
            if token == row['tokenized_span'][0]:
                ner_tags[idx] = 'B'
                for i in range(1, len(row['tokenized_span'])):
                    if idx + i < len(row['tokenized_text']) and row['tokenized_text'][idx + i] == row['tokenized_span'][i]:
                        ner_tags[idx + i] = 'I'
                    else:
                        break
        return ner_tags
    
    train_df['ner_tags'] = train_df.apply(create_ner_tags, axis=1)
    val_df['ner_tags'] = val_df.apply(create_ner_tags, axis=1)
    test_df['ner_tags'] = test_df.apply(create_ner_tags, axis=1)

    # Reanme columns
    train_df = train_df.rename(columns={
        'tokenized_text': 'tokens',
    })
    val_df = val_df.rename(columns={
        'tokenized_text': 'tokens',
    })
    test_df = test_df.rename(columns={
        'tokenized_text': 'tokens',
    })

    if args.extraction_type == 'discriminative':
        # Change the labels to B-ANTECEDENT and I-ANTECEDENT
        train_df['ner_tags'] = train_df['ner_tags'].apply(lambda x: [label_list[0] if label == 'O' else label_list[1] if label == 'B' else label_list[2] for label in x])
        val_df['ner_tags'] = val_df['ner_tags'].apply(lambda x: [label_list[0] if label == 'O' else label_list[1] if label == 'B' else label_list[2] for label in x])
        test_df['ner_tags'] = test_df['ner_tags'].apply(lambda x: [label_list[0] if label == 'O' else label_list[1] if label == 'B' else label_list[2] for label in x])
    
    elif args.extraction_type == 'extractive':
        # Change the labels to B-{label} and I-{label}
        train_df['ner_tags'] = train_df.apply(
            lambda row: [label_list[0] if label == 'O' else f"B-{row['elliptical_type']}" if label == 'B' else f"I-{row['elliptical_type']}"  for label in row['ner_tags']],
            axis=1
        )
        val_df['ner_tags'] = val_df.apply(
            lambda row: [label_list[0] if label == 'O' else f"B-{row['elliptical_type']}" if label == 'B' else f"I-{row['elliptical_type']}"  for label in row['ner_tags']],
            axis=1
        )
        test_df['ner_tags'] = test_df.apply(
            lambda row: [label_list[0] if label == 'O' else f"B-{row['elliptical_type']}" if label == 'B' else f"I-{row['elliptical_type']}"  for label in row['ner_tags']],
            axis=1
        )
    
    else:
        raise ValueError('Extraction type must be "discriminative" or "extractive"')

    # Print the first example
    def print_example(words, labels):
        line1 = ""
        line2 = ""
        for word, label in zip(words, labels):
            max_length = max(len(word), len(label))
            line1 += word + " " * (max_length - len(word) + 1)
            line2 += label + " " * (max_length - len(label) + 1)

        print(line1)
        print(line2)
        print('--'*20)

    print_example(train_df['tokens'].iloc[0], train_df['ner_tags'].iloc[0])
    print_example(val_df['tokens'].iloc[0], val_df['ner_tags'].iloc[0])
    print_example(test_df['tokens'].iloc[0], test_df['ner_tags'].iloc[0])

    # Load dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # # Tokenize dataset
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["candidate_text"],
            padding='max_length', 
            truncation=True, 
            max_length=256,
            return_token_type_ids=True,
        )
        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            label_ids = []
            for token_idx, token_id in enumerate(tokenized_inputs['input_ids'][i]):
                if token_id < 4: # CLS and SEP, PAD tokens
                    label_ids.append(-100)
                else:
                    label_ids.append(label2id[label[token_idx-1]])
            labels.append(label_ids)

        tokenized_inputs["labels"] = torch.tensor(labels)

        return tokenized_inputs

    tokenized_datasets = {
        name: dataset.map(tokenize_and_align_labels, batched=True)
        for name, dataset in {"train": train_dataset, "validation": val_dataset, "test": test_dataset}.items()
    }

    # Define label list
    metric = evaluate.load("seqeval")

    # Define compute metrics function
    def compute_metrics(p):
        predictions, labels = p
        predictions = predictions.argmax(axis=2)

        true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
        predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        for i in range(min(len(true_labels), 20)):
            print('Model Predictions:')
            print(print_example(tokenized_datasets['validation']['tokens'][i], predictions[i]))
            print('True Labels:')
            print(print_example(tokenized_datasets['validation']['tokens'][i], true_labels[i]))
            print('--'*20)

        results = metric.compute(predictions=predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy='no',
        load_best_model_at_end=False,
        eval_steps=args.eval_steps,
        logging_dir=args.output_dir,
        seed=args.seed,
        report_to=args.report_to if args.use_wandb else None,
        push_to_hub=args.push_to_hub,
        learning_rate=args.learning_rate,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()