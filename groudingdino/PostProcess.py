from .utils.utils import sigmoid, get_phrases_from_posmap_np, create_positive_map_from_span
import numpy as np

def xywhnorm2xyxy(ctx, cty, w, h, W, H):
    return [(ctx-w/2)*W, (cty-h/2)*H, (ctx+w/2)*W, (cty+h/2)*H]

class PostProcess():
    def __init__(self, token_spans, tokenizer, box_threshold, text_threshold, with_logits=True):
        self.tokenizer = tokenizer
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.with_logits = with_logits
        self.positive_maps = None

    def __call__(self, caption, output, W, H):
        # get phrase
        self.tokenized = self.tokenizer(caption)
        logits = sigmoid(output[0][0]) # (nq, 256)
        boxes = output[1][0]  # (nq, 4)

        # filter output
        logits_filt = logits
        boxes_filt = boxes

        filt_mask = np.max(logits_filt,axis=1) > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        
        # build pred
        pred_phrases = []
        res_boxes = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap_np(logit > self.text_threshold, self.tokenized, self.tokenizer)
            if self.with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
            ctx, cty, w, h = box
            res_boxes.append(xywhnorm2xyxy(ctx, cty, w, h, W, H))
        return res_boxes, pred_phrases