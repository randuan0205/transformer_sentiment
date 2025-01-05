import os
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset, Dataset
from gensim.utils import simple_preprocess
from gensim import corpora, downloader
from transformers import BertTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, Split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc, log_loss

# Basic setup
# from google.colab import drive
# drive.mount('/content/drive')
# import os
# os.chdir('drive/MyDrive/oldAIboy/firstAIproject')

# Import dependencies
import os
import pandas as pd
import numpy as np

# Import for data preparation
import pickle
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Section 1 build tokens
from gensim.utils import simple_preprocess
from gensim import corpora
# Section 2, create tokenizer based on hugging face, and build the dataloaders
from gensim import downloader
from transformers import BertTokenizer
from datasets import load_dataset, Split
from datasets import Dataset
# Section 4 & 5
import torch
import torch.nn as nn
from transformer_modules.transformer import *
from transformer_modules.stepbystep_v5 import StepByStep

# Pull in the data
rawdata = pd.read_csv('IMDB Dataset.csv')
print('finish testing print here')

# Section 1 build dictionary stoi (this will handle transforming strings to indices)
sentlist = rawdata['review'].tolist()
words = [simple_preprocess(sent) for sent in sentlist]  # use gensim.simple_preprocess to make tokens
stoi = corpora.Dictionary(words)  # dictionary is a set of unique tokens, in the format of gansim dict

# Create data label
dataset = Dataset.from_pandas(rawdata)
def create_label(row):
    sentiment_label = int(row['sentiment'] == 'positive')
    return {'labels': sentiment_label}
dataset = dataset.map(create_label)

# Show one example of the movie review data
print(dataset[0])

# Data Preparation 2 - build tokenizer by using GLOVE and huggingface
max_len = 300
batch_size = 32

# Section 2, create tokenizer based on GLOVE by using huggingface
glove = downloader.load('glove-wiki-gigaword-50')
# Save embedding's vocab to a plain txt file, then utilize HF's tokenizer
to_add = ['[PAD]', '[UNK]']
words = glove.index_to_key
words = to_add + words
folder = os.getcwd()
# Only need to run this once - create the local txt file
# with open(os.path.join(folder, 'vocab2.txt'), 'w') as f:
#     for word in words:
#         f.write(f'{word}\n')

# Create the tokenizer based on local txt file
glove_tokenizer = BertTokenizer('vocab2.txt')

# Data Preparation 3 - Implement the text tokenization and create data batches through PyTorch's DataLoader
with open("data_tokens_300seq.pickle", "rb") as f:
    train_ids, train_labels, test_ids, test_labels = pickle.load(f)

# Create data batch by using DataLoader
generator = torch.Generator()
train_tensor_dataset = TensorDataset(train_ids, train_labels)
train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True, generator=generator)
test_tensor_dataset = TensorDataset(test_ids, test_labels)
test_loader = DataLoader(test_tensor_dataset, batch_size=batch_size)

# Data Preparation 4 - Create the embedding layer by using GLOVE (An pretrained text embedding)
special_embeddings = np.zeros((2, glove.vector_size))
extended_embeddings = np.concatenate([special_embeddings, glove.vectors], axis=0)

# Loads the pretrained GloVe embeddings into an embedding layer
extended_embeddings = torch.as_tensor(extended_embeddings).float()
torch_embeddings = nn.Embedding.from_pretrained(extended_embeddings)

# Model 1 - design and architecture
class TransfClassifier(nn.Module):
    def __init__(self, embedding_layer, encoder, n_outputs):
        super().__init__()
        self.d_model = encoder.d_model
        self.n_outputs = n_outputs
        self.encoder = encoder
        self.mlp = nn.Linear(self.d_model, n_outputs)

        self.embed = embedding_layer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

    def preprocess(self, X):
        # N, L -> N, L, D
        src = self.embed(X)
        # Special classifier token
        # 1, 1, D -> N, 1, D
        cls_tokens = self.cls_token.expand(X.size(0), -1, -1)
        # Concatenates CLS tokens -> N, 1 + L, D
        src = torch.cat((cls_tokens, src), dim=1)
        return src

    def encode(self, source, source_mask=None):
        # Encoder generates "hidden states"
        states = self.encoder(source, source_mask)
        # Gets state from first token only: [CLS]
        cls_state = states[:, 0]  # N, 1, D
        return cls_state

    @staticmethod
    def source_mask(X):
        cls_mask = torch.ones(X.size(0), 1).type_as(X)
        pad_mask = torch.cat((cls_mask, X > 0), dim=1).bool()
        return pad_mask.unsqueeze(1)

    def forward(self, X):
        src = self.preprocess(X)
        # Featurizer
        cls_state = self.encode(src, self.source_mask(X))
        # Classifier
        out = self.mlp(cls_state)  # N, 1, outputs
        return out

# Model 2 - model training & evaluation (200 epochs / lr 1e-5)
n_layers = 4
n_heads = 5
ff_units = 128

# Train the model
torch.manual_seed(33)
# Creates a Transformer Encoder
layer = EncoderLayer(n_heads=n_heads, d_model=torch_embeddings.embedding_dim, ff_units=ff_units)
encoder = EncoderTransf(layer, n_layers=n_layers, max_len=max_len)

# Uses both layers above to build our model
model = TransfClassifier(torch_embeddings, encoder, n_outputs=1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

sbs_transf = StepByStep(model, loss_fn, optimizer)
sbs_transf.set_loaders(train_loader, test_loader)
sbs_transf.train(201, print_interval=20, patience=8, min_delta=0)

fig = sbs_transf.plot_losses()

# Model evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

logits_tensor = torch.empty((0, 1), device=device)  # Create an empty tensor on the device

# Loop through the test data in batches of 100
for i in range(0, len(test_ids), 100):
    # Extract a batch of 100 data points
    batch = test_ids[i:i + 100].to(device)

    # Perform model inference
    with torch.no_grad():  # Disable gradient calculation during inference
        logits = sbs_transf.model(batch)

    # Append the logits to the list
    logits_tensor = torch.cat([logits_tensor, logits], dim=0)  # Concatenate logits to the tensor

# Evaluate model performance
# Assuming you have:
# - logits_tensor: a PyTorch tensor containing the inference logits
# - test_labels: a PyTorch tensor containing the true labels

# 1. Convert logits to probabilities
probabilities = torch.sigmoid(logits_tensor).cpu().detach().numpy()

# 2. Convert probabilities to binary predictions (0 or 1)
predictions = (probabilities >= 0.5).astype(int)

# 3. Calculate AUC score
auc_score = roc_auc_score(test_labels.cpu().detach().numpy(), probabilities)

# 4. Calculate accuracy score
accuracy = accuracy_score(test_labels.cpu().detach().numpy(), predictions)

# 5. Calculate ROC curve
fpr, tpr, thresholds = roc_curve(test_labels.cpu().detach().numpy(), probabilities)
roc_auc = auc(fpr, tpr)

# 6. Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# 7. Calculate logged loss
# Convert test_labels to NumPy and ensure it's in the correct shape
test_labels_np = test_labels.cpu().detach().numpy()
# test_labels_np = test_labels_np.reshape(-1, 1)  # Reshape if necessary (e.g., if it's a 1D array)
# test_labels_np = test_labels_np.astype(int)  # Ensure data type is int

# Clip probabilities to avoid log(0) errors
probabilities_clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)

logged_loss = log_loss(test_labels_np, probabilities_clipped)

# Print the results
print(f"AUC Score: {auc_score}")
print(f"Accuracy Score: {accuracy}")
print(f"Logged Loss: {logged_loss}")

# Model 3 - Model save, load, inference
# Save the model - Assuming you have:
# - model: your trained PyTorch model
# - PATH: the file path where you want to save the model (e.g., 'my_model.pth')

# 1. Save the model's state_dict
torch.save(sbs_transf.model.state_dict(), 'deep_model_200epochs_trained1227.pth')

# Load the model and implement inference
# Set up hyper-parameters
n_layers = 4
n_heads = 5
ff_units = 128

# Load the model
# Assuming you have:
# - model: an instance of the same model class that you saved
# - PATH: the file path where the model is saved

# 1. Load the saved state_dict
state_dict = torch.load('deep_model_200epochs_trained1227.pth')

# 2. Load the state_dict into your model
torch.manual_seed(33)
# Loads the pretrained GloVe embeddings into an embedding layer
extended_embeddings = torch.as_tensor(extended_embeddings).float()
torch_embeddings = nn.Embedding.from_pretrained(extended_embeddings)
# Creates a Transformer Encoder
layer = EncoderLayer(n_heads=n_heads, d_model=torch_embeddings.embedding_dim, ff_units=ff_units)
encoder = EncoderTransf(layer, n_layers=n_layers, max_len=max_len)
# Uses both layers above to build our model
model = TransfClassifier(torch_embeddings, encoder, n_outputs=1)

model.load_state_dict(state_dict)

# 3. Set the model to evaluation mode (if needed)
model.eval()

# Implement the model inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

logits_tensor = torch.empty((0, 1), device=device)  # Create an empty tensor on the device

# Loop through the test data in batches of 100
for i in range(0, len(test_ids), 100):
    # Extract a batch of 100 data points
    batch = test_ids[i:i + 100].to(device)

    # Perform model inference
    model.to(device)
    with torch.no_grad():  # Disable gradient calculation during inference
        logits = model(batch)

    # Append the logits to the list
    logits_tensor = torch.cat([logits_tensor, logits], dim=0)  # Concatenate logits to the tensor

# Section 3. Trying to understand self-attention mechanism by looking into the attention matrix
# 3.1 example1 - a positive example
reviewno = 2
print(dataset[reviewno])

# Obtain the tokens
tokens = glove_tokenizer(dataset['review'][reviewno], truncation=True, padding=True, max_length=max_len-1, add_special_tokens=False, return_tensors='pt')['input_ids']

# Perform model inference & produce the probability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
with torch.no_grad():  # Disable gradient calculation during inference
    logits = model(tokens.to(device))
print(f'logits are {logits}')

# Check the attention matrix matched to the review texts
decoded_text = glove_tokenizer.decode(tokens[0])
attscores = model.encoder.layers[3].self_attn_heads.alphas.squeeze()[:,:,0]

# Split decoded_text into words
words = decoded_text.split()

# Create a list to store the data for the DataFrame
data = []

# Assuming 'attscores' has 5 dimensions
num_dimensions = attscores.shape[0]

# Check if the number of dimensions matches the expected number
if num_dimensions != 5:
    print(f"Warning: attscores has {num_dimensions} dimensions, expected 5. Using available dimensions")

# Iterate through the words and create rows for the DataFrame
for i, word in enumerate(words):
    if i < len(attscores[0]):  # Check to make sure we don't go out of bounds
        row_data = [word]  # First column is the current word
        for j in range(min(5, num_dimensions)):  # Iterate through the attention dimensions
            row_data.append(attscores[j][i].item())  # Append the attention score of the current word for the current attention head
        data.append(row_data)
    else:
        break  # Stop if we reach the end of the attention scores

# Create the DataFrame
columns = ['word']
for i in range(min(5, num_dimensions)):
    columns.append(f'head_{i+1}')  # Change dimension_n to head_n

df = pd.DataFrame(data, columns=columns)
print(df)

# Filter rows where any value in 'head_1' to 'head_5' is greater than 0.008
scorethresh = 0.008
df_filtered = df[(df['head_1'] > scorethresh) | (df['head_2'] > scorethresh) | (df['head_3'] > scorethresh) | (df['head_4'] > scorethresh) | (df['head_5'] > scorethresh)]
print(df_filtered)

# 3.2 example2 - a negative example
reviewno = 3
print(dataset[reviewno])

# Obtain the tokens
tokens = glove_tokenizer(dataset['review'][reviewno], truncation=True, padding=True, max_length=max_len-1, add_special_tokens=False, return_tensors='pt')['input_ids']

# Perform model inference & produce the probability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
with torch.no_grad():  # Disable gradient calculation during inference
    logits = model(tokens.to(device))
print(f'logits are {logits}')

# Assuming 'decoded_text' and 'attscores' are defined from the previous code
decoded_text = glove_tokenizer.decode(tokens[0])
attscores = model.encoder.layers[3].self_attn_heads.alphas.squeeze()[:,:,0]

# Split decoded_text into words
words = decoded_text.split()

# Create a list to store the data for the DataFrame
data = []

# Assuming 'attscores' has 5 dimensions
num_dimensions = attscores.shape[0]

# Check if the number of dimensions matches the expected number
if num_dimensions != 5:
    print(f"Warning: attscores has {num_dimensions} dimensions, expected 5. Using available dimensions")

# Iterate through the words and create rows for the DataFrame
for i, word in enumerate(words):
    if i < len(attscores[0]):  # Check to make sure we don't go out of bounds
        row_data = [word]  # First column is the current word
        for j in range(min(5, num_dimensions)):  # Iterate through the attention dimensions
            row_data.append(attscores[j][i].item())  # Append the attention score of the current word for the current attention head
        data.append(row_data)
    else:
        break  # Stop if we reach the end of the attention scores

# Create the DataFrame
columns = ['word']
for i in range(min(5, num_dimensions)):
    columns.append(f'head_{i+1}')  # Change dimension_n to head_n

df = pd.DataFrame(data, columns=columns)
print(df)

# Filter rows where any value in 'head_1' to 'head_5' is greater than 0.013
scorethresh = 0.013
df_filtered = df[(df['head_1'] > scorethresh) | (df['head_2'] > scorethresh) | (df['head_3'] > scorethresh) | (df['head_4'] > scorethresh) | (df['head_5'] > scorethresh)]
print(df_filtered)
