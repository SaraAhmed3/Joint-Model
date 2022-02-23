import torch
import eval
from sklearn.metrics import f1_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metric(dataloader, decoder):
    
    total_acc_final = 0
    pred_shape = []
    pred_size = []
    pred_color = []
    pred_action = []
    pred_intent = []
    
    true_shape = []
    true_size = []
    true_color = []
    true_action = []
    true_intent = []
    for iter, sample_batched in enumerate(dataloader):
        
        total_acc = 0
        
        shape, size, color, action, intent, hidden_state, last_hidden = sample_batched

        predicted_shape, predicted_size, predicted_color, predicted_action, predicted_intent = \
                    eval.evaluate(last_hidden, hidden_state, decoder)
        
        # Batch_size = hidden_state.size()[0]
        pred_shape.extend(predicted_shape.squeeze())
        pred_size.extend(predicted_size.squeeze())
        pred_color.extend(predicted_color.squeeze())
        pred_action.extend(predicted_action.squeeze())
        pred_intent.extend(predicted_intent.squeeze())
  
        true_shape.extend(shape)
        true_size.extend(size)
        true_color.extend(color)
        true_action.extend(action)
        true_intent.extend(intent)
    shape_score = f1_score(true_shape, pred_shape, average= 'macro')
    size_score = f1_score(true_size, pred_size, average= 'macro')
    color_score = f1_score(true_color, pred_color, average= 'macro')
    action_score = f1_score(true_action ,pred_action, average= 'macro')
    intent_score = f1_score(true_intent, pred_intent, average= 'macro')
    return shape_score, size_score, color_score, action_score, intent_score