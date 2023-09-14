from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer
from transformers import AutoModel

app = Flask(__name__)

class SentimentModel(torch.nn.Module):
    def __init__(self, num_labels):
        super(SentimentModel, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def predict_sentiment(model, tokenizer, input_sentence):
    model.eval()

    # Tokenize và mã hóa câu nhập vào
    encoding = tokenizer.encode_plus(
        input_sentence,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.sigmoid(logits)
        predicted_labels = predicted_labels = torch.round(probabilities).cpu().int()

    predicted_label_names = []
    for label in predicted_labels:
        label_names = [target_names[i] for i, value in enumerate(label) if value == 1]
        predicted_label_names.extend(label_names)

    return predicted_label_names

def replace_synonyms(sentence, synonym_dict):
    words = sentence.split()
    replaced_words = []

    for word in words:
        cnt = 0
        for i, values in synonym_dict.iterrows():
            if word in list(map(lambda x : x.lstrip().rstrip(), values["Synonym"].split(","))):          
                replaced_words.append(values["Word"])
                cnt = 1
        if cnt == 1:
            continue
        else:
            replaced_words.append(word)
    return " ".join(replaced_words)

target_names = ['positive', 'negative', 'neutral', 'toxic', 'confused', 'funny', 'admirable', 'compassionate']

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = SentimentModel(num_labels=8)
model.load_state_dict(torch.load('sentiment_model.pth', map_location=torch.device('cpu')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for processing the input and returning the predicted label
import pandas as pd
synonym_dict = pd.read_excel(r'D:\HK4\DS103-thu-thap-va-tien-xu-ly-du-lieu\demo\Quy đổi từ đồng nghĩa  (1).xlsx')
@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['sentence']
    replaced_sentence = replace_synonyms(sentence, synonym_dict)
    predicted_labels = predict_sentiment(model, tokenizer, replaced_sentence)
    return render_template('index.html', sentence=sentence, labels=predicted_labels)

if __name__ == '__main__':
    app.run(debug=True)
