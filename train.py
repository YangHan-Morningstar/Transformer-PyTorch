from transformer.Models import Transformer
from transformer.Optim import TransformerOptimizer
from transformer.utils import create_masks
from utils import collate_fn, SelfDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实际训练时要预先划分好训练、验证、测试集，想路径传递到相应的DataLoader
source_filepath = "./simulated_data"

train_dataset = SelfDataset(source_filepath)
train_data_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

val_dataset = SelfDataset(source_filepath)
val_data_loader = DataLoader(
    val_dataset,
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
last_val_epoch_loss = float("inf")
for i in range(epochs):
    epoch_loss = 0
    loop = tqdm(train_data_loader)
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

        loop.set_description("Training Epoch {}".format(i + 1))
        loop.set_postfix(loss=batch_loss.item())
    print("Training Epoch {}: Loss: {:.4}".format(i + 1, epoch_loss / len(train_data_loader)))

    model.eval()
    with torch.no_grad():
        val_epoch_loss = 0
        loop = tqdm(val_data_loader)
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

            val_epoch_loss += batch_loss.item()

            loop.set_description("Val Epoch {}".format(i + 1))
            loop.set_postfix(loss=batch_loss.item())
        average_val_epoch_loss = val_epoch_loss / len(val_data_loader)
        print("Val Epoch {}: Loss: {:.4}".format(i + 1, average_val_epoch_loss))

        if average_val_epoch_loss < last_val_epoch_loss:
            last_val_epoch_loss = average_val_epoch_loss
            torch.save(model.state_dict(), "./trained_weights/{:.4}.pth".format(last_val_epoch_loss))
            print("最优模型权重已保存")
        else:
            print("当前不是最优模型，未保存")

