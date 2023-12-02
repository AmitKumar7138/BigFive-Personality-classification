import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import preprocess
import os
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction


class RoBERTaPersonalityTrainer:
    def __init__(self, dataframe, model_name='roberta-base', output_dir='RoBerta_small', num_labels=5, epochs=3, lr=2e-5, batch_size=16):
        self.dataframe = dataframe
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_labels = num_labels
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.dataframe['TEXT'] = self.dataframe['TEXT'].apply(preprocess)
        self.split_data()

    def split_data(self):
        labels = self.dataframe[['cEXT', 'cNEU',
                                 'cAGR', 'cCON', 'cOPN']].values
        X_train, X_val, y_train, y_val = train_test_split(
            self.dataframe['TEXT'], labels, test_size=0.2)
        self.X_train, self.X_val = X_train.reset_index(
            drop=True).tolist(), X_val.reset_index(drop=True).tolist()
        self.y_train, self.y_val = y_train.tolist(), y_val.tolist()

    class PersonalityDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=512,
                                                padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
            input_ids = inputs['input_ids'].flatten()
            attention_mask = inputs['attention_mask'].flatten()
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}

    def multi_label_metrics(self, predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(
            y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy}
        return metrics

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions
        result = self.multi_label_metrics(
            predictions=preds,
            labels=p.label_ids)
        return result

    def train_and_evaluate(self):
        train_dataset = self.PersonalityDataset(
            self.X_train, self.y_train, self.tokenizer)
        val_dataset = self.PersonalityDataset(
            self.X_val, self.y_val, self.tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, problem_type="multi_label_classification", num_labels=self.num_labels)
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=self.epochs,
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

        model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        eval_results = trainer.evaluate()
        predictions = trainer.predict(val_dataset)
        predicted_probs = torch.sigmoid(
            torch.from_numpy(predictions.predictions)).numpy()
        print(predicted_probs)
        predicted_labels = (predicted_probs > 0.5).astype(int)
        y_val_array = np.array(self.y_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_val_array, predicted_labels)

        # Print evaluation results including accuracy
        print("\nEvaluation Results:")
        for key, value in eval_results.items():
            print(f"{key}: {value}")
        print(f"Accuracy: {accuracy}")

        # Convert each row in the arrays into a separate list inside a larger list
        y_values_list = y_val_array.tolist()
        predicted_values_list = predicted_labels.tolist()

        # Create pandas Series from these lists of lists
        y_values_series = pd.Series(y_values_list)
        predicted_values_series = pd.Series(predicted_values_list)

        # Create DataFrame
        df_vals = pd.DataFrame({
            'Y_values': y_values_series,
            'predicted_values': predicted_values_series
        })

        # Save the DataFrame as a CSV file
        df_vals.to_csv(
            f'RoBERTa_output{self.batch_size}{self.lr}{self.epochs}.csv', index=False)
