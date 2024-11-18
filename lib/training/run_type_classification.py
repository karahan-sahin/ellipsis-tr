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
from lib.training.t5encoder import T5EncoderForSequenceClassification
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

    parser.add_argument('--over_sample', action='store_true', help='Over sample the minority classes')

    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')

    parser.add_argument('--min_count_per_class', type=int, default=None, help='Minimum count per class')
    parser.add_argument('--max_count_per_class', type=int, default=None, help='Maximum count per class')

    parser.add_argument('--train_size', type=str, default='all', help='Train size')
    parser.add_argument('--dev', action='store_true', help='Development mode')

    return parser.parse_args()

if __name__ == "__main__":
    # Load the CSV file into a pandas DataFrame
    args = parse_args()

    if 'bert' in args.model_name:
        model_name = 'berturk'
    elif 'mt5' in args.model_name:
        model_name = 'mt5'
    elif 'TURNA' in args.model_name:
        model_name = 'turna'
    else:
        raise ValueError(f"Model {args.model_name} is not supported")
    
    if 'release' in args.dataset_file:
        model_name = f'{model_name}-release'
    if 'challenge' in args.dataset_file:
        model_name = f'{model_name}-challenge'

    # create a wandb run_name
    run_name = f"{model_name}-{args.model_type}-{args.train_size}-type-classification-mn={args.min_count_per_class}-mx={args.max_count_per_class}-os={args.over_sample}"
    init_wandb(run_name, args)

    train_df = pd.read_csv(args.dataset_file)
    val_df = pd.read_csv(args.dataset_file.replace('train', 'val'))
    test_df = pd.read_csv(args.dataset_file.replace('train', 'test'))

    print('Train Data Before Filtering:')
    print(train_df['elliptical_type'].value_counts().to_markdown())

    if args.min_count_per_class is not None:
        train_counts = train_df.value_counts('elliptical_type').reset_index(name='count') 
        train_df = train_df[train_df['elliptical_type'].isin(train_counts[train_counts['count'] >= args.min_count_per_class]['elliptical_type'])]

        # If a class is removed from the train set, remove it from the val and test sets as well
        val_df = val_df[val_df['elliptical_type'].isin(train_df['elliptical_type'])]
        test_df = test_df[test_df['elliptical_type'].isin(train_df['elliptical_type'])]
        
        print('Train Data After Filtering:')
        print(train_df['elliptical_type'].value_counts().to_markdown())
        print('*'*20)
        print('Val Data After Filtering:')
        print(val_df['elliptical_type'].value_counts().to_markdown())
        print('*'*20)
        print('Test Data After Filtering:')
        print(test_df['elliptical_type'].value_counts().to_markdown())
        print('*'*20)

    if args.max_count_per_class is not None:
        # lİMİT THE NUMBER OF EXAMPLES PER CLASS
        train_df = train_df.groupby('elliptical_type').head(args.max_count_per_class)
        val_df = val_df.groupby('elliptical_type').head(args.max_count_per_class)
        test_df = test_df.groupby('elliptical_type').head(args.max_count_per_class)

    if args.over_sample:
        # Oversample the minority classes
        def oversample_minority_classes(df, max_majority_count=2000):
            from sklearn.utils import resample
            majority_count = df['elliptical_type'].value_counts().max()
            # First limit the number of examples in the majority class
            if max_majority_count is not None:
                majority_count = min(majority_count, max_majority_count)
                df[df['elliptical_type'] == df['elliptical_type'].value_counts().idxmax()].sample(majority_count)
                
            minority_classes = df['elliptical_type'].value_counts().index[1:]
            for minority_class in minority_classes:
                minority_df = df[df['elliptical_type'] == minority_class]
                minority_df = resample(minority_df, replace=True, n_samples=majority_count)
                df = pd.concat([df, minority_df])

            # Shuffle the DataFrame
            df = df.sample(frac=1).reset_index(drop=True)

            return df
        
        print('Train Data Before Over Sampling:')
        print(train_df['elliptical_type'].value_counts().to_markdown())
        print('*'*20)

        train_df = oversample_minority_classes(train_df)
        
        print('Train Data After Over Sampling:')
        print(train_df['elliptical_type'].value_counts().to_markdown())
        print('*'*20)

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

    # Tokenize the dataset
    train_df['tokens'] = train_df['text'].apply(lambda x: tokenizer.tokenize(x))
    val_df['tokens'] = val_df['text'].apply(lambda x: tokenizer.tokenize(x))
    test_df['tokens'] = test_df['text'].apply(lambda x: tokenizer.tokenize(x))

    # print num_examples, avg_length per dataset
    def print_dataset_stats(df, name):
        num_examples = len(df)
        avg_length = df['tokens'].apply(len).mean()
        print(f'{name} dataset: {num_examples} examples, avg_length: {avg_length:.2f}')


    print_dataset_stats(train_df, 'Train')
    print_dataset_stats(val_df, 'Val')
    print_dataset_stats(test_df, 'Test')

    if args.model_type == 'encoder':

        if model_name == 'turna' or model_name == 'mt5':
            model = T5EncoderForSequenceClassification(args.model_name, num_labels)

        else:
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
            save_strategy='steps',
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            save_total_limit=1,
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
