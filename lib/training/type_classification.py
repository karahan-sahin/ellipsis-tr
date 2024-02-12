
import os
import logging
from datasets import DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers import PushToHubCallback
from transformers import Trainer, TrainingArguments

from lib.eval.compute_metrics import compute_metrics_for_type_classification
from lib.dataset.dataset import EllipsisType, HumanAnnotationStatus, Environment
from lib.dataset.augment import add_augmented_data

logger = logging.getLogger(__name__)

def train(model: PreTrainedModel, 
          train_dataset: DatasetDict, 
          eval_dataset: DatasetDict, 
          tokenizer: PreTrainedTokenizerFast, 
          data_collator: function, 
          compute_metrics: function) -> None:
    """Train the model on the given dataset. 

    Args: 
        model (PreTrainedModel): The model to be trained. 
        train_dataset (DatasetDict): The training dataset.
        eval_dataset (DatasetDict): The evaluation dataset.
        tokenizer (PreTrainedTokenizerFast): The tokenizer for the model.
        data_collator (function): The data collator for the model.
        compute_metrics (function): The function to compute the metrics for the model.
        
    Returns:
        None
    """
    
    
    MODEL_NAME = f'ellipsis-tr-{MODEL}-{TASK_TYPE}-{EXP_TYPE}'
    
    training_args = TrainingArguments(
        output_dir=f"{os.environ['MODEL_OUTPUT_DIR']}/{MODEL_NAME}",
        learning_rate=os.environ['LR'],
        per_device_train_batch_size=os.environ['BATCH_SIZE'],
        per_device_eval_batch_size=os.environ['BATCH_SIZE'],
        num_train_epochs=os.environ['NUM_EPOCHS'],
        logging_steps=2,
        weight_decay=os.environ['DECAY'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=None,
        push_to_hub=True,
    )
    
    push_to_hub_callback = PushToHubCallback(
        output_dir=os.environ['MODEL_OUTPUT_DIR'], 
        tokenizer=tokenizer, 
        hub_model_id=f"karahan-sahin/{MODEL_NAME}"
    )
    
    compute_metrics = compute_metrics_for_type_classification()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            push_to_hub_callback
        ]
    )
    
    trainer.train()