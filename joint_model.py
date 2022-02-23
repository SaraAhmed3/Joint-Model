import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size_shape, output_size_size, output_size_color,
                 output_size_action, output_size_type, dropout_p=0.1,batch_size= BATCH_SIZE):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = 20
        self.dropout_p = dropout_p
        
        # self.embedding_shape = nn.Embedding(self.output_size, 204)
        # self.embedding_size = nn.Embedding(self.output_size, 204)
        # self.embedding_color = nn.Embedding(self.output_size, 204)
        # self.embedding_action = nn.Embedding(self.output_size, 204)
        # self.embedding_type = nn.Embedding(self.output_size, 208)
        
        self.attn = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
      
        self.out_shape = nn.Linear(self.hidden_size, output_size_shape)
        self.out_size = nn.Linear(self.hidden_size, output_size_size)
        self.out_color = nn.Linear(self.hidden_size, output_size_color)
        self.out_action = nn.Linear(self.hidden_size, output_size_action)
        self.out_type = nn.Linear(self.hidden_size, output_size_type)
#input_shape, input_size, input_color, input_action, input_type,
    def forward(self, hidden, cell_state, encoder_outputs):
        
        self.batch_size = encoder_outputs.size()[0]
  

      
        # embedded_color = self.embedding_color(input_color).view(1, self.batch_size, 1, -1)
        # embedded_color = self.dropout(embedded_color)
        
        # embedded_shape = self.embedding_shape(input_shape).view(1, self.batch_size, 1, -1)
        # embedded_shape = self.dropout(embedded_shape)
        
        # embedded_size = self.embedding_size(input_size).view(1, self.batch_size, 1, -1)
        # embedded_size = self.dropout(embedded_size)
        
        # embedded_mat = self.embedding_mat(input_mat).view(1, self.batch_size, 1, -1)
        # embedded_mat = self.dropout(embedded_mat)
        
        # embedded_shape = self.embedding_shape(input_shape).view(1, self.batch_size, 1, -1)
        # embedded_shape = self.dropout(embedded_shape)
        
        # embedded_size = self.embedding_size(input_size).view(1, self.batch_size, 1, -1)
        # embedded_size = self.dropout(embedded_size)
        
        # embedded_color = self.embedding_color(input_color).view(1, self.batch_size, 1, -1)
        # embedded_color = self.dropout(embedded_color)
        
        # embedded_action = self.embedding_action(input_action).view(1, self.batch_size, 1, -1)
        # embedded_action = self.dropout(embedded_action)
        
        # embedded_type = self.embedding_type(input_type).view(1, self.batch_size, 1, -1)
        # embedded_type = self.dropout(embedded_type)
        
       
        # embedded = torch.cat((embedded_shape[0], embedded_size[0], embedded_color[0], embedded_action[0], embedded_type[0]), 2)
         
        # hidden: batch_sizex1x1024, encoder_outputs: batch_sizex93x1024
        
        attn_align = F.tanh((hidden + encoder_outputs))    
        

        attn_weights = F.softmax((self.attn(attn_align)).view(self.batch_size, 1, -1) , dim=-1)
        
        attn_applied = torch.bmm(attn_weights, encoder_outputs) #.unsqueeze(0))
       

        output = attn_applied
       
        output, (hidden, cell_state) = self.lstm(output, (hidden.view(1,self.batch_size,-1), cell_state))

        output = output.view(self.batch_size,-1)

        output_shape = F.log_softmax(self.out_shape(output), dim=1)
        output_size = F.log_softmax(self.out_size(output), dim=1) 
        output_color = F.log_softmax(self.out_color(output), dim=1)
        output_action = F.log_softmax(self.out_action(output), dim=1)
        output_type = F.log_softmax(self.out_type(output), dim=1)
               
        
        return output_shape, output_size, output_color, output_action, output_type, hidden, cell_state, attn_weights   