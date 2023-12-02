import os
import re
import pandas as pd
import numpy as np
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(y_true, y_pred):
    '''
    Purpose: Computes evaluation metrics (accuracy, precision, recall, and F1 score) for binary classification tasks.
    Input:
    y_true: An array of true labels.
    y_pred: An array of predicted labels.
    Output: Returns a tuple containing accuracy, precision, recall, and f1 scores, all rounded to four decimal places.
    '''
    # Flatten the arrays if they are multidimensional (for multi-label tasks)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    accuracy = accuracy_score(y_true, y_pred).round(4)
    precision = precision_score(
        y_true, y_pred, average='binary', zero_division=0).round(4)
    recall = recall_score(y_true, y_pred, average='binary',
                          zero_division=0).round(4)
    f1 = f1_score(y_true, y_pred, average='binary').round(4)

    return accuracy, precision, recall, f1


def string_to_list(s):
    '''
    Purpose: Safely converts a string representation of a list into an actual list. This function is typically used for parsing strings back into list objects.
    Input: s: A string representation of a list.
    Output: Returns a list object if the conversion is successful; otherwise, returns an empty list.
    '''
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []


def compute_metrics(PATH_FOLDER):
    # New DataFrame to store results
    '''
    Purpose: Computes and aggregates the evaluation metrics for multiple CSV files (each representing a different model's performance) within a specified directory. The function assumes that the filename contains specific model parameters.
    Input: PATH_FOLDER: Path to the folder containing the CSV files.
    Output: A DataFrame results_df with columns for model_name, batch_size, learning_rate, epochs, trait, accuracy, precision, recall, and f1-score. It includes the computed metrics for each trait ('cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN') of each model.
    '''
    results_df = pd.DataFrame(columns=['model_name', 'batch_size', 'learning_rate',
                                       'epochs', 'trait', 'accuracy', 'precision', 'recall', 'f1-score'])

    folder_path = PATH_FOLDER  # Replace with the path to your folder

    # Regular expression pattern to extract parameters from filename
    pattern = r'(\D+)(\d*)(\D*)(\d+).csv'

    # Process each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            match = re.match(pattern, filename)
            if match:
                # Extract parameters
                model_name, batch_size, learning_rate, epochs = match.groups()
                batch_size = int(
                    batch_size) if batch_size.isdigit() else batch_size
                learning_rate = float(learning_rate) if learning_rate.replace(
                    '.', '', 1).isdigit() else learning_rate
                epochs = int(epochs)

            # Read the file
            df = pd.read_csv(os.path.join(folder_path, filename))
            df['Y_values'] = df['Y_values'].apply(string_to_list)
            df['predicted_values'] = df['predicted_values'].apply(
                string_to_list)

            y_true = np.array(df['Y_values'].tolist())
            y_predicted = np.array(df['predicted_values'].tolist())

            # Compute metrics for each trait
            # Compute metrics for each trait
            for i, trait in enumerate(['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']):
                y = y_true[:, i]
                p = y_predicted[:, i]
                accuracy, precision, recall, f1 = compute_metrics(y, p)

                # Append results to the DataFrame
                new_row = pd.DataFrame({'model_name': model_name,
                                        'batch_size': batch_size,
                                        'learning_rate': learning_rate,
                                        'epochs': epochs,
                                        'trait': trait,
                                        'accuracy': accuracy,
                                        'precision': precision,
                                        'recall': recall,
                                        'f1-score': f1}, index=[0])
                results_df = pd.concat(
                    [results_df, new_row], ignore_index=True)

                new_row = pd.DataFrame({'model_name': model_name,
                                        'batch_size': batch_size,
                                        'learning_rate': learning_rate,
                                        'epochs': epochs,
                                        'trait': trait,
                                        'accuracy': accuracy,
                                        'precision': precision,
                                        'recall': recall,
                                        'f1-score': f1}, index=[0])
                results_df = pd.concat(
                    [results_df, new_row], ignore_index=True)

    return results_df


def refined_metrirc_csv(df):
    '''
    # Function to refine metrics CSV for BERT and RoBERTa models
    # Input: DataFrame containing raw metrics data.
    # Output: Refined DataFrame with modified learning_rate, batch_size, epochs, and dropped unnecessary columns.
    '''
    for i in range(len(df['batch_size'])):
        inc = str(df['batch_size'][i])[-1]
        dec = str(df['epochs'][i])[0]
        df['learning_rate'][i] = f"{inc}{df['learning_rate'][i]}{dec}"
        df['batch_size'][i] = str(df['batch_size'][i])[:-1]
        df['epochs'][i] = str(df['epochs'][i])[-1]
    df = df.drop(['accuracy', 'precision', 'recall'], axis=1)
    return df


def transform_df(df2):
    '''
    # Function to transform the DataFrame
    # Input: DataFrame with model performance metrics for each trait in subsequent rows.
    # Output: Transformed DataFrame with a single row for each model consolidating all trait scores and average F1 score.
    '''
    df_new = pd.DataFrame(columns=['model_name', 'batch_size',
                                   'learning_rate', 'epochs', 'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN', 'f1-average'])
    # List to store the new rows
    new_rows = []
    # Iterate over df2 with a step of 5
    for i in range(0, len(df2), 5):
        new_row = {
            'model_name': df2.at[i, 'model_name'],
            'batch_size': df2.at[i, 'batch_size'],
            'learning_rate': df2.at[i, 'learning_rate'],
            'epochs': df2.at[i, 'epochs'],
            'cEXT': df2.at[i, 'f1-score'],
            'cNEU': df2.at[i+1, 'f1-score'] if i+1 < len(df2) else None,
            'cAGR': df2.at[i+2, 'f1-score'] if i+2 < len(df2) else None,
            'cCON': df2.at[i+3, 'f1-score'] if i+3 < len(df2) else None,
            'cOPN': df2.at[i+4, 'f1-score'] if i+4 < len(df2) else None,
            'f1-average': ((df2.at[i, 'f1-score']+df2.at[i+1, 'f1-score']+df2.at[i+2, 'f1-score']+df2.at[i+3, 'f1-score']+df2.at[i+4, 'f1-score'])/5).round(4)
        }
        new_rows.append(new_row)
    df_new = pd.DataFrame(new_rows)

    return df_new


def evaluate_performance_nonTransformer(df_path_list):
    '''
    # Function to evaluate performance of non-Transformer models
    # Input: List of file paths for non-Transformer model results.
    # Output: DataFrame with F1 scores for each trait and the average F1 score for each model.
    '''

    results_df = pd.DataFrame(
        columns=['model_name', 'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN', 'f1-average'])
    for item in df_path_list:
        df = pd.read_csv(item)
        name = item.split('_')[0]
        df['Y_values'] = df['Y_values'].apply(string_to_list)
        df['predicted_values'] = df['predicted_values'].apply(string_to_list)

        y_true = np.array(df['Y_values'].tolist())
        y_predicted = np.array(df['predicted_values'].tolist())

        # Compute metrics for each trait
        f1_score = []
        for i, trait in enumerate(['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']):
            y = y_true[:, i]
            p = y_predicted[:, i]
            _, _, _, f1 = compute_metrics(y, p)
            f1_score.append(f1)

        new_row = pd.DataFrame({'model': name,
                                'cEXT': f1_score[0],
                                'cNEU': f1_score[1],
                                'cAGR': f1_score[2],
                                'cCON': f1_score[3],
                                'cOPN': f1_score[4],
                                'average_f1': sum(f1_score)/5}, index=[0])

        results_df = pd.concat([results_df, new_row], ignore_index=True)

    return results_df


def get_best_scores(PATH):
    '''
    # Function to get best scores
    # Input: DataFrame with model performance metrics.
    # Output: Two DataFrames - one for the best models by average F1 score, and another for the best scores per trait.
    '''
    best_f1 = pd.DataFrame(columns=['model_name', 'batch_size', 'learning_rate',
                                    'epochs', 'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN', 'f1-average'])

    best_traits = pd.DataFrame(columns=['model_name', 'batch_size', 'learning_rate',
                                        'epochs', 'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN', 'f1-average'])

    df = pd.read_csv(PATH)

    BERT_small = df[(df['model_name'] == 'BERT_small_output') & (
        df['f1-average'] == df[df['model_name'] == 'BERT_small_output']['f1-average'].max())]
    BERT_Large = df[(df['model_name'] == 'BERT_Large_output') & (
        df['f1-average'] == df[df['model_name'] == 'BERT_Large_output']['f1-average'].max())]
    RoBERTa_small = df[(df['model_name'] == 'RoBERTa_output') & (
        df['f1-average'] == df[df['model_name'] == 'RoBERTa_output']['f1-average'].max())]
    RoBERTa_Large = df[(df['model_name'] == 'RoBERTa_Large_output') & (
        df['f1-average'] == df[df['model_name'] == 'RoBERTa_Large_output']['f1-average'].max())]

    f1 = [BERT_small, BERT_Large, RoBERTa_small, RoBERTa_Large]
    for item in f1:
        best_f1 = pd.concat([best_f1, item], ignore_index=True)

    best_f1.to_csv('best_f1.csv', index=False)

    cEXT_max = df[df['cEXT'] == df['cEXT'].max()]
    cNEU_max = df[df['cNEU'] == df['cNEU'].max()]
    cAGR_max = df[df['cAGR'] == df['cAGR'].max()]
    cCON_max = df[df['cCON'] == df['cCON'].max()]
    cOPN_max = df[df['cOPN'] == df['cOPN'].max()]

    traits = [cEXT_max, cNEU_max, cAGR_max, cCON_max, cOPN_max]
    for item in traits:
        best_traits = pd.concat([best_traits, item], ignore_index=True)

    best_traits.to_csv('best_traits.csv', index=False)


# Concatinating the resutls for best f1_score for each model non-Transformer and Transformer
df = pd.read_csv('final_results.csv')
df2 = pd.read_csv('best_f1.csv')


final_df = pd.concat([df, df2[['model_name', 'cEXT', 'cNEU',
                     'cAGR', 'cCON', 'cOPN', 'f1-average']]], ignore_index=True)
final_df.to_csv('final_results.csv', index=False)
