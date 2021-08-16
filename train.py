from transformer.Models import Transformer
from transformer.Optim import TransformerOptimizer
from transformer.utils import create_masks
from utils import collate_fn, SelfDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source_filepath = "./simulated_data"
dataset = SelfDataset(source_filepath)
data_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

model = Transformer(
    input_vocab_size=6210,
    target_vocab_size=5233
).to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

optimizer = TransformerOptimizer(
    optimizer=Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
    d_model=512
)

epochs = 1000

for i in range(epochs):
    epoch_loss = 0
    loop = tqdm(data_loader)
    model.train()
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

        batch_loss.backward()
        optimizer.step_and_update_lr()
        optimizer.zero_grad()

        epoch_loss += batch_loss.item()

        loop.set_description("EPOCH {} BATCH {}".format(i + 1, batch_i + 1))
        loop.set_postfix(loss=batch_loss.item())
