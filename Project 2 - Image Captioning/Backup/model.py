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
        super(DecoderRNN, self).__init__()
        
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.vocab_size = vocab_size
#         self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout = 0, batch_first = True)
        #self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size)) 
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        #nn.init.xavier_uniform_(self.fc.weight)
        #nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        captions = self.embedding(captions)
        features = features.unsqueeze(dim=1)
        
        inputs = torch.cat((features, captions), dim=1)
        
        outputs, self.hidden = self.lstm(inputs)
        outputs = self.fc(outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output_sentence = []
        #output_lenght = 0
        
        #while output_lenght < max_len:
        for i in range(max_len):
            output, hidden = self.lstm(inputs, states)
            #print('LSTM Output: ' + str(output))
            
            output = self.fc(output.squeeze(1))
            #print('FC Output: ' + str(output))
            #_, output = output.max(1)
            
            _, word_index = torch.max(output, 1) #output.max(1)[1]
            #print('Word: ' + str(output.item()))
            
            output_sentence.append(word_index.cpu().numpy()[0].item())
            if word_index == 1: 
                break
            inputs = self.embedding(word_index)
            inputs = inputs.unsqueeze(1)
            #inputs = inputs.unsqueeze(1)
            #output_lenght = output_lenght + 1
            
        return output_sentence
    