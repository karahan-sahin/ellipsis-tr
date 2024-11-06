import os
import pandas as pd
from datasets import Dataset
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoTokenizer, 
    AutoModelForSequenceClassification
)

from dotenv import load_dotenv
load_dotenv()

def init_wandb(run_name):
    import wandb
    wandb.init(
        project="ellipsis-tr",
        entity="boun-pilab",
        name=run_name,
    )

def parse_args():

    import argparse
    parser = argparse.ArgumentParser(description='Train a model for text classification')
    
    # Required arguments
    parser.add_argument('--dataset_file', type=str, required=True, help='Path to the CSV file containing the data')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')

    # Optional arguments
    parser.add_argument('--model_type', type=str, default='encoder' , help='Model type')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the model')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for training')
    parser.add_argument('--evaluation_strategy', type=str, default='epoch', help='Evaluation strategy')
    parser.add_argument('--logging_steps', type=int, default=1, help='Logging steps')
    parser.add_argument('--report_to', type=str, default=None, help='Report to')
    parser.add_argument('--push_to_hub', action='store_true', help='Push to Hub')
    parser.add_argument('--hub_model_id', type=str, default=None, help='Hub model ID')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')

    return parser.parse_args()


if __name__ == "__main__":
    # Load the CSV file into a pandas DataFrame
    args = parse_args()

    # create a wandb run_name
    run_name = f"{args.model_name.split('/')[-1]}-{args.dataset_file.split('/')[-1].split('.')[0]}"
    init_wandb(run_name)

    train_df = pd.read_csv(args.dataset_file)
    val_df = pd.read_csv(args.dataset_file.replace('train', 'val'))
    test_df = pd.read_csv(args.dataset_file.replace('train', 'test'))
    
    # Rename columns {'candidate_text': 'text', }
    train_df = train_df.rename(columns={'candidate_text': 'text', })
    val_df = val_df.rename(columns={'candidate_text': 'text', })
    test_df = test_df.rename(columns={'candidate_text': 'text', })

    # Get the unique labels
    labels = train_df['elliptical_type'].unique()
    num_labels = len(labels)

    # Create label2id and id2label mappings
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in id2label.items()}

    # Format labels as integers
    train_df['label'] = train_df['elliptical_type'].apply(lambda x: label2id[x])
    val_df['label'] = val_df['elliptical_type'].apply(lambda x: label2id[x])
    test_df['label'] = test_df['elliptical_type'].apply(lambda x: label2id[x])
    
    # Convert the DataFrame to a Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.model_type == 'encoder':
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, 
            num_labels=num_labels, 
            id2label=id2label,
        )
        # Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", truncation=True)
        
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
        tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

        # Create EarlyStoppingCallback
        from transformers import EarlyStoppingCallback
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="steps",
            logging_steps=args.logging_steps,

            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_epochs,
            weight_decay=args.weight_decay,

            report_to=args.report_to,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            hub_token=os.environ.get('HF_TOKEN', None),
            load_best_model_at_end=True,
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            tokenizer=tokenizer,
            callbacks=[early_stopping],
        )

    elif args.model_type == 'encoder_decoder':
        def tokenize_function(examples):
            inputs = tokenizer(
                examples['text'], 
                text_target=examples['elliptical_type'],
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
        }

        optimizer_params = {
            'optimizer_type': 'adafactor',
            'scheduler': False,
        }

        from turkish_lm_tuner import TrainerForClassification

        model_trainer = TrainerForClassification(
            model_name=args.model_name,
            task='classification',
            training_params=training_params,
            optimizer_params=optimizer_params,
            model_save_path=args.output_dir,
            num_labels=num_labels,
        )

        trainer, model = model_trainer.train_and_evaluate(tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset)

        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    else:
        raise ValueError(f"Model type {args.model_type} is not supported")


    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)

    # Save the evaluation results
    with open('eval_results.txt', 'w') as f:
        print(eval_results, file=f)

    # Save the scores
    scores = eval_results['eval_accuracy']
    with open('scores.txt', 'w') as f:
        print(scores, file=f)