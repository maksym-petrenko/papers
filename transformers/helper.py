import torch


def positional_encoding(length: int, d_model: int):
    
    result = torch.zeros((d_model, length, 2))
    
    indicies = torch.arange(0, length).to(dtype=torch.float32)
    positions = torch.arange(0, d_model)

    positions = 0.1 ** (8 * positions / d_model).to(dtype=torch.float32) 
    positions = positions.unsqueeze(0).reshape((d_model, 1))
    indicies = indicies.unsqueeze(0)
    sheet = torch.matmul(positions, indicies)

    result[:, :, 0] = torch.sin(sheet)
    result[:, :, 1] = torch.cos(sheet)

    return result.reshape((d_model, 2, length)).flatten(start_dim=1)


def is_english_or_french(char: str) -> bool:
    code = ord(char)
    return (
        (0x0020 <= code <= 0x007E) or
        (0x00A0 <= code <= 0x00FF)
    )
