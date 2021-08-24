import torch
from transformer.Models import Transformer
from transformer.utils import create_masks
import json


def read_from_json(path):
    with open(path) as temp_file:
        json_dict = json.load(temp_file)
    return json_dict


class TransformerModel(object):

    def __init__(self, model, src_dicts, tar_dicts, max_length=100):
        super(TransformerModel, self).__init__()
        self.model = model
        self.src_dicts = src_dicts
        self.tar_dicts = tar_dicts
        self.tar_dicts_reverse = {value: key for key, value in tar_dicts.items()}

        self.src_start_token = [src_dicts["<S>"]]
        self.src_end_token = [src_dicts["<E>"]]
        self.tar_start_token = [tar_dicts["<S>"]]
        self.tar_end_token = tar_dicts["<E>"]

        self.max_length = max_length

        self.model.eval()

    def trans(self, sentence):
        sentence_idx = []
        for word in sentence:
            if word in self.src_dicts:
                sentence_idx.append(self.src_dicts[word])
            else:
                sentence_idx.append(self.src_dicts["<UNK>"])
        inp_sentence = self.src_start_token + sentence_idx + self.src_end_token
        inp_sentence = torch.tensor(inp_sentence)
        encoder_inp = torch.unsqueeze(inp_sentence, dim=0)

        decoder_inp = torch.tensor(self.tar_start_token)
        output = torch.unsqueeze(decoder_inp, dim=0)

        with torch.no_grad():
            for i in range(self.max_length):
                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                    encoder_inp,
                    output
                )

                predictions, _ = self.model(
                    inp=encoder_inp,
                    tar=output,
                    enc_padding_mask=enc_padding_mask,
                    look_ahead_mask=combined_mask,
                    dec_padding_mask=dec_padding_mask
                )

                predictions = predictions[:, -1:, :]
                predictions_id = torch.argmax(predictions, dim=-1)
                if predictions_id[0][0] == self.tar_end_token:
                    output = torch.squeeze(output)
                    break

                output = torch.cat((output, predictions_id), dim=-1)

            output = torch.squeeze(output) if len(output.size()) == 2 else output

        output = output.numpy()[1:]
        sentence_tar = "".join([self.tar_dicts_reverse[i] for i in output])
        return sentence_tar


if __name__ == '__main__':
    device = torch.device("cpu")

    src_dicts = read_from_json("your json dict")
    tar_dicts = read_from_json("your json dict")

    model = Transformer(
        input_vocab_size=len(src_dicts),
        target_vocab_size=len(tar_dicts),
        position_encoding_input=10000,
        position_encoding_target=10000,
        dropout_rate=0.5,
        device=device
    ).to(device)
    model_state_dict = torch.load(
        "your model weights filepath",
        map_location=lambda storage, loc: storage
    )
    model.load_state_dict(model_state_dict)

    transformer = TransformerModel(model, src_dicts, tar_dicts)

    while True:
        print("请输入")
        sentence = input()
        print(transformer.trans(sentence))
