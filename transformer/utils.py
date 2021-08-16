import torch


def create_padding_mask(seq):
    return (seq == 0).float().unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(seq_len):
    return (torch.triu(torch.ones((seq_len, seq_len))) == 0).transpose(0, 1).float()


def create_masks(inp, tar):

    enc_padding_mask = create_padding_mask(inp)

    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tar.size()[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


if __name__ == '__main__':
    x = torch.tensor([[7, 6, 0, 0, 0], [1, 2, 3, 0, 0]])
    y = torch.tensor([[9, 5, 2, 0, 0, 0, 0, 0], [9, 8, 7, 6, 5, 4, 3, 0]])

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(x, y)

    print()
