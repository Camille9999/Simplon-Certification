import pandas as pd
import string
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import BertTokenizer
from transformers import BertModel

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

df_train_prompt = pd.read_csv('commonlit-evaluate-student-summaries/prompts_train.csv')
df_train_summaries = pd.read_csv('commonlit-evaluate-student-summaries/summaries_train.csv')
df_train = df_train_summaries.merge(df_train_prompt, on='prompt_id')


def count_stopwords(text: str) -> int:
    stopword_list = set(stopwords.words('english'))
    words = text.split()
    stopwords_count = sum(1 for word in words if word.lower() in stopword_list)
    return stopwords_count

# Count the punctuations in the text.
# punctuation_set -> !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
def count_punctuation(text: str) -> int:
    punctuation_set = set(string.punctuation)
    punctuation_count = sum(1 for char in text if char in punctuation_set)
    return punctuation_count

# Count the digits in the text.
def count_numbers(text: str) -> int:
    numbers = re.findall(r'\d+', text)
    numbers_count = len(numbers)
    return numbers_count

# This function applies all the above preprocessing functions on a text feature.
def feature_engineer(dataframe: pd.DataFrame, feature: str = 'text') -> pd.DataFrame:
    dataframe[f'{feature}_word_cnt'] = dataframe[feature].apply(lambda x: len(x.split(' ')))
    dataframe[f'{feature}_length'] = dataframe[feature].apply(lambda x: len(x))
    dataframe[f'{feature}_stopword_cnt'] = dataframe[feature].apply(lambda x: count_stopwords(x))
    dataframe[f'{feature}_punct_cnt'] = dataframe[feature].apply(lambda x: count_punctuation(x))
    dataframe[f'{feature}_number_cnt'] = dataframe[feature].apply(lambda x: count_numbers(x))
    return dataframe


preprocessed_df = feature_engineer(df_train)


class TextDataset(Dataset):
    def __init__(self, texts, questions, summaries, grades):
        self.texts = texts
        self.questions = questions
        self.summaries = summaries
        self.grades = grades
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        question = self.questions[idx]
        summary = self.summaries[idx]
        grade = self.grades[idx]

        # We concatenate the text, question and summary and separate them with the [SEP] token
        inputs = self.tokenizer.encode_plus(
            text + ' [SEP] ' + question + ' [SEP] ' + summary,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return input_ids, attention_mask, grade


texts = preprocessed_df['prompt_text']
questions = preprocessed_df['prompt_question']
summaries = preprocessed_df['text']
grades1 = preprocessed_df['content']
grades2 = preprocessed_df['wording']

# Create datasets
dataset1 = TextDataset(texts, questions, summaries, grades1)
dataset2 = TextDataset(texts, questions, summaries, grades2)

# Create dataloaders
train_dataloader1 = DataLoader(dataset1, batch_size=4, shuffle=True)
train_dataloader2 = DataLoader(dataset2, batch_size=4, shuffle=True)


from transformers import BertModel, BertTokenizer
import torch.nn as nn

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
base_model = BertModel.from_pretrained('bert-base-uncased')

class GradePredictor(nn.Module):
    def __init__(self):
        super(GradePredictor, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(base_model.config.hidden_size, 1)  # Predicting one grade

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        out = self.dropout(outputs.pooler_output)
        return self.linear(out)

# Instantiate two separate models
model1 = GradePredictor()
model2 = GradePredictor()

# Define loss function - since we're predicting a single grade in each model, we can use Mean Squared Error (MSE)
criterion = nn.MSELoss()

# Define optimizers
optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-5)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-5)


# Number of training epochs
num_epochs = 10

for epoch in range(num_epochs):
    # Training loop for model1
    total_loss1 = 0  # Keep track of the total loss
    for batch in train_dataloader1:
        # Zero the gradients
        optimizer1.zero_grad()

        # Get input data and labels
        input_ids, attention_mask, labels = batch
        labels = labels.float()

        # Forward pass
        outputs = model1(input_ids=input_ids, attention_mask=attention_mask).squeeze()

        # Compute loss
        loss = criterion(outputs, labels)
        total_loss1 += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer1.step()

    avg_train_loss1 = total_loss1 / len(train_dataloader1)  # Calculate average loss over the entire training data
    print(f"Model 1 - Epoch: {epoch+1}, Loss: {avg_train_loss1:.2f}")

    # Training loop for model2
    total_loss2 = 0  # Keep track of the total loss
    for batch in train_dataloader2:
        # Zero the gradients
        optimizer2.zero_grad()

        # Get input data and labels
        input_ids, attention_mask, labels = batch
        labels = labels.float()

        # Forward pass
        outputs = model2(input_ids=input_ids, attention_mask=attention_mask).squeeze()

        # Compute loss
        loss = criterion(outputs, labels)
        total_loss2 += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer2.step()

    avg_train_loss2 = total_loss2 / len(train_dataloader2)  # Calculate average loss over the entire training data
    print(f"Model 2 - Epoch: {epoch+1}, Loss: {avg_train_loss2:.2f}")
