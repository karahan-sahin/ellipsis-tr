import os
import json
import numpy as np
import pandas as pd
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    Trainer, 
    TrainingArguments, 
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    DataCollatorForTokenClassification
)

from turkish_lm_tuner import T5ForClassification
from turkish_lm_tuner.trainer import BaseModelTrainer

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()

class TrainerForClassification(BaseModelTrainer):
    def __init__(self, model_name, task, training_params, optimizer_params, model_save_path, num_labels, postprocess_fn=None):
        super().__init__(model_name, training_params, optimizer_params)
        self.num_labels = num_labels
        self.task = task
        test_params = training_params.copy()

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if self.task == "semantic_similarity":
            preds = preds.flatten()
        else:
            preds = np.argmax(preds, axis=-1)

        logger.info('Postprocessing..')

        if self.task in ["ner", "pos_tagging"]:
            preds, labels = self.postprocess_fn((preds, labels))
        else:
            preds = self.postprocess_fn(preds)
            labels = self.postprocess_fn(labels)

        logger.info("Computing metrics")

        result = super().compute_metrics(preds, labels)

        logger.info("Predictions: %s", preds[:5])
        logger.info("Labels: %s", labels[:5])

        predictions = pd.DataFrame(
            {'Prediction': preds,
             'Label': labels
            })

        predictions.to_csv(os.path.join(self.test_params['output_dir'], 'predictions.csv'), index=False)

        logger.info("Result: %s", result)

        return result

    def initialize_model(self):
        config = AutoConfig.from_pretrained(self.model_name)
        if config.model_type in ["t5", "mt5"]:
            if self.task == "classification":
                return T5ForClassification(self.model_name, config, self.num_labels, "single_label_classification")
            elif self.task in ["ner", "pos_tagging"]:
                return T5ForClassification(self.model_name, config, self.num_labels, "token_classification")
            else:
                return T5ForClassification(self.model_name, config, 1, "regression")
        else:
            if self.task == "classification":
                return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
            elif self.task in ["ner", "pos_tagging"]:
                return AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
    
    def train_and_evaluate(self, train_dataset, eval_dataset, test_dataset):
        logger.info("Training in classification mode")

        if self.task in ['ner', 'pos_tagging']:
            data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
            tokenizer = self.tokenizer
        else:
            data_collator, tokenizer = None, None
        training_args = TrainingArguments(
            metric_for_best_model='eval_loss',
            load_best_model_at_end=True,
            greater_is_better=False,
            **self.training_params)
        logger.info("Training arguments: %s", training_args)

        model = self.initialize_model()
        if self.optimizer_params is not None:
            logger.info("Using optimizers with constant parameters")
            optimizer, lr_scheduler = self.create_optimizer(model)
        else:
            logger.info("Using optimizers created based on training_arguments")
            optimizer, lr_scheduler = (None, None)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            optimizers=(optimizer, lr_scheduler),
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.train()
        results = trainer.evaluate(test_dataset)
        
        logger.info("Results: %s", results)
        json.dump(results, open(os.path.join(self.training_params['output_dir'], "results.json"), "w"))
        return trainer, model