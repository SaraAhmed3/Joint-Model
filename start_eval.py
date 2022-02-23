import joint_model
import evaluation_metric
import torch
from torch import optim
import dataloader 
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())

def start_eval(checkpoint_file = None):
    hidden_size = 1024
    output_size_shape = 6
    output_size_size = 4
    output_size_color = 7
    output_size_action = 5
    output_size_type  = 3
    batch_size = 512

    attn_decoder = joint_model.AttnDecoderRNN(hidden_size, output_size_shape, output_size_size, output_size_color,
                 output_size_action, output_size_type, dropout_p=0.1).to(device)
    
    decoder_optimizer = optim.SGD(attn_decoder.parameters(), lr=0.01)

    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
        attn_decoder.load_state_dict(checkpoint['state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['optimizer'])

    attn_decoder.eval()
    
    output_folder_1 = 'D:/uOttawa/final_project/pickle_test-20220108T160947Z-001/pickle_test'   
    output_folder_2 = 'D:/uOttawa/final_project/pickle_inference'
    shapes_dataset = dataloader.ShapeDataset(pickle_dir = output_folder_2)
    shapes_dataloader = DataLoader(shapes_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return evaluation_metric.metric(shapes_dataloader, attn_decoder) 
    
