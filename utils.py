import sys
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import torch
import numpy as np
import os
from tqdm import tqdm
from transformer.utils import create_masks, compute_bleu


def collate_fn(batch_data):
    src_list, tar_inp_list, tar_real_list = [], [], []
    for i in range(len(batch_data)):
        src, tar_inp, tar_real = batch_data[i][0], batch_data[i][1], batch_data[i][2]
        src_list.append(torch.tensor(src))
        tar_inp_list.append(torch.tensor(tar_inp))
        tar_real_list.append(torch.tensor(tar_real))

    src_padding = pad_sequence(src_list, batch_first=True, padding_value=0)
    tar_inp_padding = pad_sequence(tar_inp_list, batch_first=True, padding_value=0)
    tar_real_padding = pad_sequence(tar_real_list, batch_first=True, padding_value=0)

    return src_padding, tar_inp_padding, tar_real_padding


class SelfDataset(Dataset):

    def __init__(self, source_filepath, data_name):
        self.source_filepath = source_filepath
        self.name_list = data_name

    def __getitem__(self, item):
        data = np.load(self.source_filepath + "/" + self.name_list[item])
        src, tar = data["src"], data["tar"]
        tar_inp, tar_real = tar[:-1], tar[1:]
        return [src, tar_inp, tar_real]

    def __len__(self):
        return len(self.name_list)


def train_on_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    if is_main_process():
        data_loader = tqdm(data_loader)

    for batch_i, (batch_src, batch_tar_inp, batch_tar_real) in enumerate(data_loader):

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            batch_src,
            batch_tar_inp
        )

        batch_src = batch_src.to(device)
        batch_tar_inp = batch_tar_inp.to(device)
        batch_tar_real = batch_tar_real.to(device)

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
        loss = reduce_value(batch_loss, average=True)
        mean_loss = (mean_loss * batch_i + loss.detach()) / (batch_i + 1)

        if is_main_process():
            data_loader.set_description("Training Epoch {}".format(epoch + 1))
            data_loader.set_postfix(mean_loss=round(mean_loss.item(), 4))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step_and_update_lr()
        optimizer.zero_grad()

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate_or_test(model, data_loader, device, epoch=0):
    model.eval()
    bleu_sum = torch.zeros(1).to(device)

    if is_main_process():
        data_loader = tqdm(data_loader)

    for batch_i, (batch_src, batch_tar_inp, batch_tar_real) in enumerate(data_loader):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            batch_src,
            batch_tar_inp
        )

        batch_src = batch_src.to(device)
        batch_tar_inp = batch_tar_inp.to(device)
        batch_tar_real = batch_tar_real.to(device)

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

        prediction_ids = torch.argmax(predictions, dim=-1)
        prediction_ids = torch.squeeze(prediction_ids, dim=0).cpu().numpy()[:-1]
        y_true = torch.squeeze(batch_tar_real, dim=0).cpu().numpy()[:-1]
        bleu_sum += compute_bleu(y_true, prediction_ids)

        if is_main_process():
            data_loader.set_description("Evaluating or Testing Epoch {}".format(epoch + 1))

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    bleu_sum = reduce_value(bleu_sum, average=False)

    return bleu_sum.item()


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value


def read_from_json(path):
    with open(path) as temp_file:
        temp_dict = json.load(temp_file)
    return temp_dict
