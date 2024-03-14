import argparse
import numpy as np
import random
import os
import cv2
from npuengine import EngineOV
from PIL import Image, ImageDraw, ImageFont
from transformers import BertTokenizerFast, AutoTokenizer
from utils.tools import *


class RAM:
    def __init__(self,
        swin_path='bmodel/ram_swin_f16_bm1684x.bmodel',
        tagging_head_path='bmodel/ram_tagging_head_bm1684x_f16.bmodel',
        dino_path='bmodel/groundingdino_bm1684x_fp16.bmodel',
        tag_list='resources/tag_list/ram_tag_list.txt',
        tag_list_chinese='resources/tag_list/ram_tag_list_chinese.txt',
        tokenizer_path = './groudingdino/utils/bert-base-uncased',
        device_id=0
    ):
        self.dino = EngineOV(dino_path, device_id=device_id)
        self.swin = EngineOV(swin_path, device_id=device_id)
        self.tagging_head = EngineOV(tagging_head_path, device_id=device_id)
        self.tag_list = load_tag_list(tag_list)
        self.tag_list_chinese = load_tag_list(tag_list_chinese)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    
    def __call__(self, image_path, return_bbox=True, box_threshold=0.25, text_threshold=0.20, iou_threshold=0.5):
        token_en, token_ch = self.detect_tag(image_path)
        ret = {
            'tag_en': token_en,
            'tag_ch': token_ch,
            'img_res': None
        }
        if return_bbox:
            img_with_bboxes = self.get_bbox(image_path,
                                            token_en, 
                                            box_threshold=box_threshold, 
                                            text_threshold=text_threshold, 
                                            iou_threshold=iou_threshold)
            # img_with_bboxes.save(f'{box_threshold}-{text_threshold}-{iou_threshold}.jpg')
            ret['img_res'] = img_with_bboxes
        return ret


    def detect_tag(self, image_path):
        input_swin = preprocess(image_path, image_size=(384, 384))
        output_swin = self.swin([input_swin])
        image_embeds, label_embed = output_swin[0], output_swin[1]
        image_atts = np.ones((1, 145)).astype(np.int32)

        output_tag = self.tagging_head([label_embed, image_embeds, image_atts])
        tag = output_tag[0]
        tag_output = []
        tag_output_chinese = []

        index = np.argwhere(tag[0] == 1)
        token = self.tag_list[index].squeeze(axis=1) ######## ram output, set of tags
        token_chinese = self.tag_list_chinese[index].squeeze(axis=1)
        return token, token_chinese
    
    def get_bbox(self, image_path, token, box_threshold=0.25, text_threshold=0.2, iou_threshold=0.5):
        input_dino = preprocess(image_path, image_size=(800, 800))

        caption = ', '.join(token)
        caption = caption.lower()
        caption = caption.strip()

        if not caption.endswith("."):
            caption = caption + "."
        caption = [caption] ######## DINO input
        tokenized = self.tokenizer(caption, padding="max_length", return_tensors="np",max_length=256)
        text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(tokenized)
        text_token_mask = tokenized["attention_mask"].astype(bool)
        proposals = gen_encoder_output_proposals()
        input_ids, token_type_ids, attention_mask = tokenized["input_ids"], tokenized["token_type_ids"], text_self_attention_masks

        output_dino = self.dino([input_dino, position_ids.astype(np.int32), 
                                text_self_attention_masks.astype(np.float32),
                                input_ids.astype(np.int32),
                                token_type_ids.astype(np.int32),
                                attention_mask.astype(np.float32),
                                text_token_mask.astype(np.float32),
                                proposals.astype(np.float32)])
        logits, boxes = sigmoid(output_dino[0][0]), output_dino[1][0]

        # filter output
        logits_filt = logits.copy()
        boxes_filt = boxes.copy()

        filt_mask = logits_filt.max(axis=1) > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            scores.append(np.max(logit).item())
        
        raw_image = Image.open(image_path)
        raw_image = raw_image.convert("RGB")
        
        image_draw = ImageDraw.Draw(raw_image)
        W, H = raw_image.size
        
        post_tokenized = self.tokenizer(caption)
        
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, self.tokenizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = nms(boxes_filt, scores, iou_threshold)
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")
        
        for i in range(boxes_filt.shape[0]):
            boxes_filt[i] = boxes_filt[i] * np.array([W, H, W, H]) 
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2 
            boxes_filt[i][2:] += boxes_filt[i][:2]
        for box, label in zip(boxes_filt, pred_phrases):
            draw_box(box, image_draw, label)
        return raw_image

