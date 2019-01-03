import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__(
        )
        self.hidden_size = hidden_size
        
        # embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=0.5, batch_first=True)

        # the linear layer that maps the hidden state output dimension 
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        
        # initialize the hidden state (see code below)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self, features, captions):

        # create embedded word vectors for each word in a sentence               
        captions_emb = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), captions_emb), dim=1)
        
                       
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        lstm_out, _ = self.lstm(embeddings)
        
        # get the scores for the most likely tag for a word
        output = self.linear(lstm_out)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output = []
        
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states) 
            out = self.linear(lstm_out.squeeze(1))           
            index = out.max(1)[1]
            output.append(index.item())
            if index == 1:
                break
            inputs = self.embed(index).unsqueeze(1) 
        
        return output
