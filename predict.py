import torch
from torch import nn
from torch import optim
from transformers import TransfoXLTokenizer, TransfoXLModel
import eval
import joint_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 16

color_dict = {"None": 0, "mauve":1, "black":2, "blue":3, "green":4, "white":5, "red":6}
id_color = {0:"None", 1:"mauve", 2: "black", 3: "blue", 4:"green", 5:"white", 6:"red"}    

size_dict = {"None": 0, "small":1, "normal":2, "big":3}
id_size = {0:"None", 1:"small", 2:"normal", 3:"big"}
     
shape_dict = {"None": 0,"ball":1, "cone":2, "cube":3, "cylinder":4, "torus":5}
id_shape = {0:"None", 1:"ball", 2:"cone", 3:"cube", 4:"cylinder", 5:"torus"}

action_dict = {"None": 0, "hover":1, "spin":2 ,"move":3}
id_action = {0:"None", 1:"hover", 2:"spin" ,3:"move"}

intent_dict = {"create":0, "animate":1, "modify":2}
id_intent = {0:"create", 1:"animate", 2:"modify"}
def Get_Embedding(description, max_length):
    
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    pretrained_model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
    
    encoded = tokenizer.encode(description)

    outputs = pretrained_model(torch.tensor(encoded).unsqueeze(0), output_hidden_states=True) 
    hidden_state = outputs.hidden_states[1] 
    last_hidden = outputs.last_hidden_state[:,-1,:]
    val = (max_length - list(hidden_state.shape)[1])
  
  
    
    hidden_state = nn.ConstantPad2d((0, 0, 0, val), 0)(hidden_state)

    return hidden_state.view(1,max_length,-1), last_hidden.view(1,1,-1)

def predictsingle(description, decoder):
    MAX_LENGTH = 16
    
    hidden_state, last_hidden = Get_Embedding(description, MAX_LENGTH)
    topi_shape, topi_size, topi_color, topi_action, topi_type = \
                    eval.evaluate(last_hidden, hidden_state, decoder)
    shape = id_shape.get(topi_shape.item())
    size = id_size.get(topi_size.item())
    color = id_color.get(topi_color.item())
    action = id_action.get(topi_action.item())
    intent = id_intent.get(topi_type.item())
    
    print("---------------------------")
    print("Describtion: ", description)
    print("Shape: ", shape)
    print("Size: ", size)
    print("Color: ", color)
    print("Action: ", action)
    print("Intent: ", intent)
    print("---------------------------")
    
def start_predict(description, checkpoint_file = None):
    
    hidden_size = 1024
    output_size_shape = 6
    output_size_size = 4
    output_size_color = 7
    output_size_action = 5
    output_size_type  = 3


    attn_decoder = joint_model.AttnDecoderRNN(hidden_size, output_size_shape, output_size_size, output_size_color,
                 output_size_action, output_size_type, dropout_p=0.1).to(device)
    
    decoder_optimizer = optim.SGD(attn_decoder.parameters(), lr=0.01)

    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
        attn_decoder.load_state_dict(checkpoint['state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['optimizer'])

    attn_decoder.eval()
    predictsingle(description, attn_decoder)

start_predict("Draw me a small red cube", r"E:\encoder_decoderV2\model_checkpoints\ck_pt100.pt")