import torch

def positional_encoding(length: int, d_model: int):

    result = torch.zeros((length, d_model))
    
    position = torch.arange(length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    result[:, 0::2] = torch.sin(div_term)
    result[:, 1::2] = torch.cos(div_term)

    result = result + position

    return result

