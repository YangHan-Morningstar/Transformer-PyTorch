import torch
from torch.utils.data import DataLoader
from transformer.Models import Transformer
from transformer.utils import create_masks
from utils import SelfDataset, collate_fn
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trained_weights_filepath = ""
model = Transformer(
    input_vocab_size=6210,
    target_vocab_size=5233
).to(device)
model.load_state_dict(torch.load(trained_weights_filepath))

source_filepath = ""
test_dataset = SelfDataset(source_filepath)
test_data_loader = DataLoader(
    test_dataset,
    batch_size=64,
    num_workers=4,
    collate_fn=collate_fn
)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

test_loss = 0
model.eval()
with torch.no_grad():
    loop = tqdm(test_data_loader)
    for batch_i, (batch_src, batch_tar_inp, batch_tar_real) in enumerate(loop):
        batch_src = batch_src.to(device)
        batch_tar_inp = batch_tar_inp.to(device)
        batch_tar_real = batch_tar_real.to(device)

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            batch_src,
            batch_tar_inp
        )

        enc_padding_mask = enc_padding_mask.to(device)
        combined_mask = combined_mask.to(device)
        dec_padding_mask = dec_padding_mask.to(device)

        predictions, _ = model(
            inp=batch_src,
            tar=batch_tar_inp,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask
        )

        batch_size, seq_len, target_vocab_size = predictions.size()
        pre = torch.reshape(predictions, shape=(batch_size * seq_len, target_vocab_size))
        real = torch.reshape(batch_tar_real, shape=(batch_size * seq_len, 1)).squeeze()

        batch_loss = loss_fn(pre, real)

        test_loss += batch_loss.item()

        loop.set_postfix(loss=batch_loss.item())
    print("Test: Loss: {:.4}".format(test_loss / len(test_data_loader)))
