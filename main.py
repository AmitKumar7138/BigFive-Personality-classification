import pandas as pd
from models.BERT_small import BERTPersonalityTrainer
from models.BERT_large import BERTLargePersonalityTrainer
from models.RoBERT_Large import RoBERTaLargePersonalityTrainer
from models.RoBERTa import RoBERTaPersonalityTrainer
from models.comparison_models import ModelTrainer
from preprocessing import preprocess_text, convert_labels_to_numeric, word2vec
from sklearn.model_selection import train_test_split


BATCH_SIZE = [16, 32]
EPOCHS = [2, 3, 4]
LEARNING_RATES = [2e-5, 3e-5, 4e-5, 5e-5]


# # Load data and preprocess
df = pd.read_csv('BIGFIVE_DATA.csv')
df = convert_labels_to_numeric(
    df, ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN'])

df['processed_Text'] = df['TEXT'].apply(preprocess_text)

word2vec(df)
df_w2v = pd.read_csv('Word2Vec_XGBoost_Dataset.csv')

# SVM model object creation
trainer_SVM = ModelTrainer("SVM")
X, y = trainer_SVM.preprocess_data(df_w2v)

# XGBoost model object creation
trainer_XGBoost = ModelTrainer("XGBoost")
X, y = trainer_XGBoost.preprocess_data(df_w2v)

# RF model object creation
trainer_RF = ModelTrainer("RF")
X, y = trainer_RF.preprocess_data(df_w2v)


for epoch in EPOCHS:
    for batch in BATCH_SIZE:
        for lr in LEARNING_RATES:
            # BERTsmall object creation
            model_BERTsmall = BERTPersonalityTrainer(
                df, epochs=epoch, batch_size=batch, lr=lr)
            model_BERTsmall.train_and_evaluate()

for epoch in EPOCHS:
    for batch in BATCH_SIZE:
        for lr in LEARNING_RATES:
            # BERTlarge object creation
            model_BERTlarge = BERTLargePersonalityTrainer(
                df, epochs=epoch, batch_size=batch, lr=lr)
            model_BERTlarge.train_and_evaluate()

for epoch in EPOCHS:
    for batch in BATCH_SIZE:
        for lr in LEARNING_RATES:
            # RoBERTa object creation
            model_RoBERTa = RoBERTaPersonalityTrainer(
                df, epochs=epoch, batch_size=batch, lr=lr)
            model_RoBERTa.train_and_evaluate()

for epoch in EPOCHS:
    for batch in BATCH_SIZE:
        for lr in LEARNING_RATES:
            # RoBERTa Large object creation
            model_RoBERTaLarge = RoBERTaLargePersonalityTrainer(
                df, epochs=epoch, batch_size=batch, lr=lr)
            model_RoBERTaLarge.train_and_evaluate()


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# # Train the models and save results
trainer_SVM.train_svm(X_train, y_train, X_test, y_test)
trainer_XGBoost.train_xgboost(X_train, y_train, X_test, y_test)
trainer_RF.train_random_forest(X_train, y_train, X_test, y_test)
