import os
import evaluate
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
)
from lib.training.turna import TrainerForClassification

from dotenv import load_dotenv
load_dotenv()

def init_wandb(run_name, args):

    if args.dev:
        os.environ['WANDB_DISABLED'] = 'true'
        return

    import wandb
    wandb.login(
        key=os.getenv('WANDB_API_KEY')
    )
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
    parser.add_argument('--learning_rate', type=float, default=2e-7, help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for training')
    parser.add_argument('--evaluation_strategy', type=str, default='steps', help='Evaluation strategy')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluation steps')
    parser.add_argument('--logging_steps', type=int, default=1, help='Logging steps')

    parser.add_argument('--push_to_hub', action='store_true', help='Push to Hub')
    parser.add_argument('--hub_model_id', type=str, default=None, help='Hub model ID')

    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')

    parser.add_argument('--train_size', type=str, default='all', help='Train size')
    parser.add_argument('--dev', action='store_true', help='Development mode')

    return parser.parse_args()

if __name__ == "__main__":
    # Load the CSV file into a pandas DataFrame
    args = parse_args()

    # create a wandb run_name
    run_name = f"{args.model_name.split('/')[-1]}-{args.dataset_file.split('/')[-1].split('.')[0]}"
    init_wandb(run_name, args)

    train_df = pd.read_csv(args.dataset_file)
    val_df = pd.read_csv(args.dataset_file.replace('train', 'val'))
    test_df = pd.read_csv(args.dataset_file.replace('train', 'test'))

    # Fil
    high_count = [
        "Genitive Drop",
        "Subject Drop",
        "Gapping",
        "Argument Drop",
        "Object Drop",
        "NP Drop",
        "No Ellipsis",
    ]

    mid_count = [
        "Genitive Drop",
        "Subject Drop",
        "Gapping",
        "Argument Drop",
        "Object Drop",
        "Stripping",
        "NP Drop",
        "Fragment",
        "Ki Expression",
        "No Ellipsis",
        "VP Ellipsis",
        "Object CP Drop"
    ]

    if args.train_size == 'high_count':
        train_df = train_df[train_df['elliptical_type'].isin(high_count)]
        val_df = val_df[val_df['elliptical_type'].isin(high_count)]
        test_df = test_df[test_df['elliptical_type'].isin(high_count)]
    elif args.train_size == 'mid_count':
        train_df = train_df[train_df['elliptical_type'].isin(mid_count)]
        val_df = val_df[val_df['elliptical_type'].isin(mid_count)]
        test_df = test_df[test_df['elliptical_type'].isin(mid_count)]
    
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
            return tokenizer(examples['text'], padding=True, truncation=True)
        
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
        tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

        # get precision, recall, f1
        accuracy = evaluate.load('accuracy')
        precision, recall, f1 = evaluate.load('precision'), evaluate.load('recall'), evaluate.load('f1')

        def compute_metrics(eval_pred):
            predictions, label_ids = eval_pred
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            predictions = predictions.argmax(axis=1)

            # Create confusion matrix
            report = classification_report(
                y_true=label_ids, y_pred=predictions, target_names=labels
            )

            print(report)
            print('--'*20)

            # Calculate the metrics
            return {
                'accuracy': accuracy.compute(
                    predictions=predictions, references=label_ids
                )['accuracy'],
                'precision': precision.compute(
                    predictions=predictions, references=label_ids, average='macro'
                )['precision'],
                'recall': recall.compute(
                    predictions=predictions, references=label_ids, average='macro'
                )['recall'],
                'macro-f1': f1.compute(
                    predictions=predictions, references=label_ids, average='macro'
                )['f1'],
                'micro-f1': f1.compute(
                    predictions=predictions, references=label_ids, average='micro'
                )['f1'],
            }

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="steps",
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,

            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,

            report_to='wandb' if not args.dev else None,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            hub_token=os.environ.get('HF_TOKEN', None),
            load_best_model_at_end=False,
            save_strategy='no'
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

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
        

    elif args.model_type == 'encoder_decoder':
        def tokenize_function(examples):
            inputs = tokenizer(
                examples['text'], 
                padding="max_length", 
                truncation=True
            )
            inputs['labels'] = [label2id[label] for label in examples['elliptical_type']]
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
