import torch
from torch.utils.data import DataLoader
from transformer.Models import Transformer
from utils import SelfDataset, collate_fn, read_from_json, evaluate_or_test
import numpy as np


device = torch.device("cpu")

src_dicts = read_from_json("./dicts/src_word_dict.json")
tar_dicts = read_from_json("./dicts/tar_word_dict.json")

trained_weights_filepath = ""
model = Transformer(
    input_vocab_size=len(src_dicts),
    target_vocab_size=len(tar_dicts),
    position_encoding_input=10000,
    position_encoding_target=10000,
    device=device
).to(device)
model.load_state_dict(
    torch.load(
        trained_weights_filepath,
        map_location=lambda storage, loc: storage
    )
)

source_filepath = ""
split_data_file = ""
data = np.load(split_data_file)
test_dataset = SelfDataset(source_filepath, data["test"])
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    num_workers=4,
    collate_fn=collate_fn
)

bleu_sum = evaluate_or_test(model=model, data_loader=test_loader, device=device)
mean_bleu = bleu_sum / len(test_loader)

print("test bleu: {:.4}".format(mean_bleu))
