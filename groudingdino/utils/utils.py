from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import os

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H

        box = box * np.array([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

import cv2
import numpy as np

def plot_boxes_to_image_cv(image_cv2, tgt):
    H, W, _ = image_cv2.shape
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have the same length"

    # Create a copy of the image to avoid modifying the original
    image_with_boxes = image_cv2.copy()

    # Draw boxes and masks
    for box, label in zip(boxes, labels):
        # From 0..1 to 0..W, 0..H
        box = box * np.array([W, H, W, H])
        # From xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # Random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # Draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        # Draw rectangle
        cv2.rectangle(image_with_boxes, (x0, y0), (x1, y1), color, 6)

        # Draw filled rectangle with label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(str(label), font, 1, 2)[0]
        cv2.rectangle(image_with_boxes, (x0, y0), (x0 + text_size[0], y0 - text_size[1]), color, -1)
        cv2.putText(image_with_boxes, str(label), (x0, y0), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image_with_boxes

def get_phrases_from_posmap_np(
    posmap, tokenized, tokenizer, left_idx: int = 0, right_idx: int = 255
):
    if len(posmap.shape) == 1:
        posmap[0: left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = np.nonzero(posmap)[0].tolist()
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        result = tokenizer.decode(token_ids)
        return result
    else:
        raise NotImplementedError("posmap must be 1-dim")
    
def generate_masks_with_special_tokens_and_transfer_map(tokenized, 
                                                        special_tokens_list=[101,102,1012,1029]):
    """Generate attention mask between each pair of special tokens
    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        torch.Tensor: attention mask between each special tokens.
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(bs, 1, 1)
    )
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)

    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )
            c2t_maski = torch.zeros((num_token), device=input_ids.device).bool()
            c2t_maski[previous_col + 1 : col] = True
        previous_col = col
    return attention_mask, position_ids.to(torch.long)

def gen_encoder_output_proposals():
    N_, S_, C_ = 1, 13294, 256
    proposals = []
    _cur = 0
    memory_padding_mask = torch.zeros(1,13294).bool()
    spatial_shapes = torch.tensor([[100, 100], [50, 50], [25, 25], [13, 13]])
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device="cpu"),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device="cpu"),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_

    output_proposals = torch.cat(proposals, 1)
    return output_proposals

def create_positive_map_from_span(tokenized, token_span, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j
    Input:
        - tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        - token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
    for j, tok_list in enumerate(token_span):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            if os.environ.get("SHILONG_DEBUG_ONLY_ONE_POS", None) == "TRUE":
                positive_map[j, beg_pos] = 1
                break
            else:
                positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
