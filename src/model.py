import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class SiameseLSTM(nn.Module):
    """
    Implements a Siamese Network using LSTMs.
    Best used with dataset mode='separate'.
    Reference: Project Guidelines suggest RNN vs Transformer comparison.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1):
        super(SiameseLSTM, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. LSTM Layer (Shared Weight Encoder)
        # batch_first=True because your dataset returns (batch, seq_len)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # 3. Classifier Layer
        # Input is hidden_dim * 4 because:
        # We have 2 inputs (q1, q2) and Bidirectional LSTM (2 directions)
        # We will concatenate: [q1_rep, q2_rep, |q1-q2|, q1*q2] for rich features
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )

    def forward_one(self, input_ids):
        # input_ids shape: (batch_size, seq_len)
        embedded = self.embedding(input_ids) # (batch, seq, embed_dim)
        
        # LSTM output: (batch, seq, directions*hidden)
        # hidden/cell: (layers*directions, batch, hidden)
        output, (hidden, cell) = self.lstm(embedded)
        
        # We take the last hidden state of the forward and backward direction
        # hidden[-2] is forward last, hidden[-1] is backward last
        final_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        return final_hidden

    def forward(self, batch_data):
        # 1. Unpack the dictionary from your dataset
        q1_ids = batch_data['q1_input_ids']
        q2_ids = batch_data['q2_input_ids']
        
        # 2. Encode both questions using the SAME LSTM (Siamese architecture)
        u = self.forward_one(q1_ids)
        v = self.forward_one(q2_ids)
        
        # 3. Combine features [u, v, |u-v|, u*v]
        # This captures both the representation and the difference/similarity
        diff = torch.abs(u - v)
        prod = u * v
        concat_features = torch.cat([u, v, diff, prod], dim=1)
        
        # 4. Classify
        logits = self.fc(concat_features)
        return logits

class BertClassifier(nn.Module):
    """
    Wrapper for a HuggingFace Transformer.
    Best used with dataset mode='concat'.
    """
    def __init__(self, model_name='bert-base-uncased', output_dim=1):
        super(BertClassifier, self).__init__()
        
        # Load pre-trained base
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freezing weights (optional: unfreeze for better performance but slower training)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
            
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, batch_data):
        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        token_type_ids = batch_data['token_type_ids']
        
        # Pass through BERT
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids)
        
        # Extract the [CLS] token representation (first token)
        cls_rep = outputs.pooler_output
        
        logits = self.fc(cls_rep)
        return logits