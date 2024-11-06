import os
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset, load_metric
from lib.training.turna import TrainerForClassification
from dotenv import load_dotenv
load_dotenv()

def init_wandb(run_name):
    import wandb
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
    parser.add_argument('--logging_steps', type=int, default=100, help='Logging steps')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16, help='Per device train batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16, help='Per device eval batch size')
    parser.add_argument('--save_steps', type=int, default=500, help='Save steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('--report_to', type=str, default='wandb', help='Report to')
    parser.add_argument('--push_to_hub', action='store_true', help='Push to hub')

    parser.add_argument('--use_wandb', action='store_true', help='Use wandb')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    if args.use_wandb:
        # Create a new run name
        run_name = f"{args.model_name}-{args.dataset_name}-{args.extraction_type}-span-classification"
        init_wandb(run_name)


    # Load dataset from csv file
    train_df = pd.read_csv(args.dataset_file)
    val_df = pd.read_csv(args.dataset_file.replace('train', 'val'))
    test_df = pd.read_csv(args.dataset_file.replace('train', 'test'))

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

    # Reanme columns
    train_df = train_df.rename(columns={
        'tokenized_text': 'tokens',
        'discriminative_span' if args.extraction_type == 'discriminative' else 'extractive_span': 'ner_tags'
    })
    val_df = val_df.rename(columns={
        'tokenized_text': 'tokens',
        'discriminative_span' if args.extraction_type == 'discriminative' else 'extractive_span': 'ner_tags'
    })
    test_df = test_df.rename(columns={
        'tokenized_text': 'tokens',
        'discriminative_span' if args.extraction_type == 'discriminative' else 'extractive_span': 'ner_tags'
    })

    # Convert the DataFrame to a Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    labels = ['Eksilti'] if args.extraction_type == 'discriminative' else train_df.elliptical_type.unique()
    num_labels = len(labels)

    def tokenize_function(examples):
        inputs = tokenizer(
            [f'ner {text}' for text in examples['candidate_text']], 
            text_target=[(f'Eksilti: {text}' if args.extraction_type == 'discriminative' else f'{ellipsis}: {text}' ) for text, ellipsis in zip(examples['span'], examples['elliptical_type'])],
            padding="max_length", 
            truncation=True
        )
        return inputs

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    training_params = {
        'num_train_epochs': args.num_epochs,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'per_device_eval_batch_size': args.per_device_eval_batch_size,
        'output_dir': args.output_dir,
        'evaluation_strategy': 'epoch',
        'save_strategy': 'epoch',   
        'predict_with_generate': True
    }

    optimizer_params = {
        'optimizer_type': 'adafactor',
        'scheduler': False,
    }

    model_trainer = TrainerForClassification(
        model_name=args.model_name,
        task='ner',
        training_params=training_params,
        optimizer_params=optimizer_params,
        model_save_path=args.output_dir,
        num_labels=num_labels,
    )

    trainer, model = model_trainer.train_and_evaluate(tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)