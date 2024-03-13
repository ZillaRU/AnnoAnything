import argparse
import os
from .utils.utils import generate_masks_with_special_tokens_and_transfer_map, gen_encoder_output_proposals, plot_boxes_to_image

import numpy as np
from transformers import BertTokenizerFast
from .PostProcess import PostProcess

import time
from PIL import Image
import sophon.sail as sail

import logging
logging.basicConfig(level=logging.INFO)


def load_image(image):
     # Load image using PIL
    image_pil = image.convert("RGB")
    # Resize the image
    image_pil = image_pil.resize((800, 800))
    # Convert PIL image to NumPy array
    image_np = np.array(image_pil)
    # Normalize the image manually
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np / 255.0 - mean) / std
    # Permute dimensions (transpose) to match the order (C, H, W)
    image_np = np.transpose(image_np, (2, 0, 1))
    return image_pil, image_np


class GroundingDINO():
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.debug("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)
        self.img_input_shape = self.net.get_input_shape(self.graph_name, self.input_name[0])

        self.tokenizer = BertTokenizerFast.from_pretrained('./groundingdino/utils/bert-base-uncased')
        self.token_spans = None  # args.token_spans

        self.batch_size = self.img_input_shape[0]
        if self.batch_size != 1:
            raise ValueError("GroundingDINO only support 1 batch")
        
        self.size = (self.img_input_shape[2], self.img_input_shape[3])

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # init postprocess
        self.postprocess = PostProcess(
            token_spans = None, # args.token_spans,
            tokenizer = self.tokenizer, 
            box_threshold = 0.3, # args.box_threshold, 
            text_threshold = 0.25 # args.text_threshold
        )

    def decode(self, img):
        self.img = Image.open(img) if isinstance(img, str) else img
    
    def preprocess(self, captions):
        if self.img is None:
            raise ValueError("Need to load your image using decode function first!")
        self.image_pil, samples = load_image(self.img)
        samples = samples[None, :, :, :]
        captions = captions.lower()
        captions = captions.strip()
        if not captions.endswith("."):
            captions = captions + "."
        captions = [captions]
        max_text_len = 256

        # encoder texts
        # load tokenizer()
        tokenized = self.tokenizer(captions, padding="max_length", return_tensors="pt")
        (text_self_attention_masks, position_ids,) = generate_masks_with_special_tokens_and_transfer_map(tokenized)
        if text_self_attention_masks.shape[1] > max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : max_text_len, : max_text_len
            ]
            position_ids = position_ids[:, : max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : max_text_len]

        # extract text embeddings
        tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
        tokenized_for_encoder["attention_mask"] = text_self_attention_masks
        tokenized_for_encoder["position_ids"] = position_ids

        tokenized = dict(tokenized)

        text_token_mask = tokenized["attention_mask"].bool()  # bs, 195
        input_ids = tokenized["input_ids"] 
        token_type_ids = tokenized["token_type_ids"] 
        attention_mask = tokenized_for_encoder["attention_mask"]
        proposals = gen_encoder_output_proposals()

        unpacked = [
            samples,
            position_ids.numpy(),
            text_self_attention_masks.numpy(),
            input_ids.numpy(),
            token_type_ids.numpy(),
            attention_mask.numpy(),
            text_token_mask.numpy(),
            proposals.numpy()
        ]
        # generate("GroundingDino", self, unpacked, "gd_workspace")
        return unpacked

    def __call__(self, data):
        if isinstance(data, list):
            values = data
        elif isinstance(data, dict):
            values = list(data.values())
        else:
            raise TypeError("data is not list or dict")
        data = {}
        for i in range(len(values)):
            data[self.input_name[i]] = values[i]
        output = self.net.process(self.graph_name, data)
        res = []

        for name in self.output_names:
            res.append(output[name])
        return res
