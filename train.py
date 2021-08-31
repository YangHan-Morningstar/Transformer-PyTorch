import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from transformer.Models import Transformer
from transformer.Optim import TransformerOptimizer
from utils import collate_fn, read_from_json
import numpy as np
from utils import SelfDataset, train_on_epoch, evaluate_or_test, init_distributed_mode, cleanup, dist


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--data-path", type=str, default="./data")
parser.add_argument("--split-data-file", type=str, default="./split_data_filepath.npz")
parser.add_argument("--src-dict-path", type=str, default="./dicts/src_word_dict.json")
parser.add_argument("--tar-dict-path", type=str, default="./dicts/tar_word_dict.json")
parser.add_argument("--trained_weights_path", type=str, default="./trained_weights/transformer.pth")
parser.add_argument("--init_weights_path", type=str, default="./trained_weights/init_weights.pth")
parser.add_argument("--device", default="cuda", help="device id (i.e. 0 or 0,1 or cpu)")
parser.add_argument("--world-size", default=4, type=int, help="number of distributed processes")
parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
args = parser.parse_args()

if torch.cuda.is_available() is False:
    raise EnvironmentError("not find GPU device for training.")

init_distributed_mode(args=args)

device = torch.device(args.device)

if args.rank == 0:
    print(args)
    print("Start to train and test")

data = np.load(args.split_data_file)

train_dataset = SelfDataset(args.data_path, data["train"])
val_dataset = SelfDataset(args.data_path, data["val"])

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)

train_batch_sampler = torch.utils.data.BatchSampler(
    train_sampler,
    args.batch_size,
    drop_last=False
)

nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
if args.rank == 0:
    print('Using {} dataloader workers every process'.format(nw))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    pin_memory=True,
    num_workers=nw,
    collate_fn=collate_fn
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    sampler=val_sampler,
    pin_memory=True,
    num_workers=nw,
    collate_fn=collate_fn
)

src_dicts = read_from_json("./dicts/src_word_dict.json")
tar_dicts = read_from_json("./dicts/tar_word_dict.json")

model = Transformer(
    input_vocab_size=len(src_dicts),
    target_vocab_size=len(tar_dicts),
    position_encoding_input=10000,
    position_encoding_target=10000,
    dropout_rate=0.1,
    device=device
).to(device)

if os.path.exists(args.trained_weights_path):
    trained_weights_dict = torch.load(args.trained_weights_path, map_location=device)
    model.load_state_dict(trained_weights_dict)
else:
    if args.rank == 0:
        torch.save(model.state_dict(), args.init_weights_path)
    dist.barrier()
    model.load_state_dict(torch.load(args.init_weights_path, map_location=device))

model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

optimizer = TransformerOptimizer(
    optimizer=Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
    d_model=512,
    lr_scale_for_multi_gpu=args.world_size
)


last_mean_bleu = 0.0
for epoch in range(args.epochs):
    train_sampler.set_epoch(epoch)

    train_epoch_mean_loss = train_on_epoch(model, optimizer, train_loader, device, epoch)

    bleu_sum = evaluate_or_test(model=model, data_loader=val_loader, device=device)
    mean_bleu = bleu_sum / val_sampler.total_size

    if args.rank == 0:
        print("Epoch {}: evaluate or test bleu: {:.4}".format(epoch + 1, mean_bleu))

        if last_mean_bleu < mean_bleu:
            last_mean_bleu = mean_bleu
            torch.save(
                model.module.state_dict(),
                "./trained_weights/model-{:.4}.pth".format(mean_bleu)
            )
            print("当前最优模型已保存")
        else:
            print("当前不是最优模型，未保存")

if args.rank == 0:
    if os.path.exists(args.init_weights_path) is True:
        os.remove(args.init_weights_path)

cleanup()
