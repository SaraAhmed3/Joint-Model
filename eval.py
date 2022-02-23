import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())

def evaluate(encoder_hidden, encoder_outputs, decoder):
    with torch.no_grad():
        

        Batch_size = encoder_hidden.size()[0]

        decoder_hidden = torch.tensor(encoder_hidden.view(Batch_size,1,-1), device=device) 
        cell_state = torch.zeros((1,Batch_size,1024), device=device)
        encoder_outputs = torch.tensor(encoder_outputs.view(Batch_size,16,-1), device=device)


        output_shape, output_size, output_color, output_action, output_type, decoder_hidden, cell_state, decoder_attention = decoder(
                                                    decoder_hidden, cell_state, encoder_outputs)
        
            
            
            
        _, topi_shape = output_shape.topk(1)
        _, topi_size = output_size.topk(1)
        _, topi_color = output_color.topk(1)
        _, topi_action = output_action.topk(1)
        _, topi_type = output_type.topk(1)
           
                

        return topi_shape, topi_size, topi_color, topi_action, topi_type
    