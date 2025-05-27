import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import re

FolderPath = []
FileName = []
PdfTextContent = []

# Function for the model export
def export_model_to_onnx(model, dummy_input, onnx_file_path):
    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        onnx_file_path,
        export_params=True,
        opset_version=14,
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    
    # Remove special characters but keep numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Load the JSON file
with open('L1Files.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract data into arrays
for entry in data:
    FolderPath.append(entry['FolderPath'])
    FileName.append(entry['FileName'])
    PdfTextContent.append(entry['PdfTextContent'])

# Create DataFrame
data = {
    "folder": FolderPath,
    "text": PdfTextContent
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Clean the text data
df['cleaned_text'] = df['text'].apply(clean_text)

# Encode the folder labels
label_encoder = {label: idx for idx, label in enumerate(df['folder'].unique())}
df['encoded_folder'] = df['folder'].map(label_encoder)

# Calculate the split index
split_index = int(len(df) * 0.8)

# Group the data by the 'folder' (category)
train_data = pd.DataFrame()  # Initialize empty DataFrames for train and test
test_data = pd.DataFrame()

# 80/20 split for each category
for category, group in df.groupby('folder'):
    # Shuffle the group (to randomize entries within each category)
    group = group.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate the split point (80% training, 20% testing)
    split_point = int(0.8 * len(group))
    
    # Split into training and testing sets
    train_data = pd.concat([train_data, group.iloc[:split_point]])  # First 80% for training
    test_data = pd.concat([test_data, group.iloc[split_point:]])     # Last 20% for testing

# Reset the indices of the final training and test sets
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

X_train = train_data['cleaned_text'].tolist()
y_train = train_data['encoded_folder'].tolist()
X_test = test_data['cleaned_text'].tolist()
y_test = test_data['encoded_folder'].tolist()

# Train-test split
# X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['encoded_folder'], test_size=0.2, random_state=42)

# Ask the user if they want to load the saved model
load_model = input("Do you want to load the saved model? (yes/no): ").strip().lower()

# Tokenizer and model from HuggingFace
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder),  hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3) # Dropout 0.1 to 0.5

# Create a custom Dataset class
class DocumentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'file_name': FileName[idx]  # Include the file name in the batch
        }

# Create DataLoader
train_dataset = DocumentDataset(X_train, y_train, tokenizer)
test_dataset = DocumentDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=0.00002)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if load_model == 'yes':
    try:
        model_load_path = 'bert_sequence_classification.pth'
        checkpoint = torch.load(model_load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Model loaded from {model_load_path}')
    except FileNotFoundError:
        print("Saved model file not found. Continuing without loading a model.")
else:
    print("Model not loaded. Continuing with a new model.")

# Training loop
num_epochs = 10

# Define the number of training steps
total_steps = len(train_loader) * num_epochs

# Create the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping - prevent the gradients from becoming too large
        optimizer.step()
        total_loss += loss.item()
    
    # Calculate average loss for the epoch
    epoch_loss = total_loss / len(train_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.7f}')
    
    # Step the scheduler based on the current epoch's loss
    scheduler.step(epoch_loss)

# Evaluate the model
model.eval()
correct = 0
total = 0
failed_test_cases = []
y_true = []
y_pred = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        file_names = batch['file_name']  # Get the file names from the batch

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect true and predicted labels for the classification report and confusion matrix
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        
        # Collect failed test cases
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                failed_test_cases.append(file_names[i])

print(f'Accuracy: {100 * correct / total:.2f}%')

print("\nClassification Report:\n", classification_report(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

print("Failed Test Cases:")
for test_case in failed_test_cases:
    print(test_case)

# Ask the user if they want to save the model
save_model = input("Do you want to save the trained model? (yes/no): ").strip().lower()

if save_model == 'yes':
    # Save the model
    model_save_path = 'bert_sequence_classification.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, model_save_path)
    print(f'Model saved to {model_save_path}')
    
    # Save the model mapping as JSON
    with open('folder_path_mapping.json', 'w') as f:
        json.dump(label_encoder, f)
    
    # Export the model to ONNX format
    dummy_input = tokenizer.encode_plus(
        "This is a dummy input",
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    dummy_input = {
        'input_ids': dummy_input['input_ids'].to(device),
        'attention_mask': dummy_input['attention_mask'].to(device)
    }

    # Path to save the ONNX model
    onnx_file_path = "bert_sequence_classification.onnx"
    export_model_to_onnx(model, dummy_input, onnx_file_path)
    print("\n\n")
    print(f"Model has been exported to {onnx_file_path}")
else:
    print("Model not saved.")

# Example usage
def classify_document(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        folder = list(label_encoder.keys())[list(label_encoder.values()).index(predicted.item())]
    
    return folder

def read_from_command_line():
    while True:               
        user_input = input("Enter some document text to classify (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        else:
            predicted_folder = classify_document(user_input)
            print(user_input)
            print("\n\n")
            print(f'The document should be saved in the "{predicted_folder}" folder.') 
            print("\n\n")


read_from_command_line()
