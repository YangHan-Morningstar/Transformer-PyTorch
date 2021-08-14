import torch


def create_padding_mask(seq):
    padding_mask = seq.clone()
    padding_mask[seq == 0] = 1
    padding_mask[seq != 0] = 0
    return padding_mask.unsqueeze(1).unsqueeze(2)


if __name__ == '__main__':
    x = torch.tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    mask = create_padding_mask(x)
    print(mask)
