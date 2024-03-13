import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_tag_list(tag_list_file):
        with open(tag_list_file, 'r', encoding="utf-8") as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list
    
def preprocess(image_path, image_size=(384, 384) , mean = np.array([0.485, 0.456, 0.406]), std = np.array([0.229, 0.224, 0.225])):
    image = Image.open(image_path)

    if image is None:
        raise ValueError(f"Image at path {image_path} is NOT EXISTS")
    image_pil = image.convert("RGB")

    image_pil_resize = image_pil.resize((image_size))

    # Convert the PIL image to a NumPy array
    image_np = np.array(image_pil_resize)
    image_np = (image_np / 255.0 - mean) / std
    image_np = np.transpose(image_np, (2, 0, 1))

    # Add a batch dimension to match the shape (1, C, H, W)
    image_np = np.expand_dims(image_np, axis=0)

    return image_np.astype(np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nms(boxes, scores, iou_threshold):
    if isinstance(scores, list):
        scores = np.array(scores) 
    if isinstance(boxes, list):
        boxes = np.array(boxes)
    keep = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def draw_box(box, draw, label):
    # random color
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    line_width = int(max(4, min(20, 0.006*max(draw.im.size))))
    
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color,  width=line_width)
    font_path = 'resources/DejaVuSans.ttf'
    font_size = int(max(12, min(60, 0.02*max(draw.im.size))))
    font = ImageFont.truetype(font_path, size=font_size)

    if label:
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white", font=font)

        draw.text((box[0], box[1]), label, font=font)


def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list=[101, 102, 1012, 1029]):
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = np.zeros((bs, num_token), dtype=bool)
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = np.column_stack(np.where(special_tokens_mask))

    # generate attention mask and positional ids
    attention_mask = np.eye(num_token, dtype=bool).reshape((1, num_token, num_token)).repeat(bs, axis=0)
    position_ids = np.zeros((bs, num_token), dtype=int)

    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = np.arange(0, col - previous_col)

        previous_col = col

    return attention_mask, position_ids


def get_phrases_from_posmap(posmap, tokenized, tokenizer):
    non_zero_idx = np.nonzero(posmap)[0].tolist()
    token_ids = [tokenized["input_ids"][0][i] for i in non_zero_idx]
    # if has ImportError
    # TRY:
    # export LD_PRELOAD={PATH TO THIS SO}libgomp-d22c30c5.so.1.0.0
    return tokenizer.decode(token_ids)


def gen_encoder_output_proposals():
    N, S, C = 1, 13294, 256
    proposals = []
    _cur = 0
    memory_padding_mask = np.zeros((1, 13294), dtype=bool)
    spatial_shapes = np.array([[100, 100], [50, 50], [25, 25], [13, 13]])
    
    for lvl, (H, W) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H * W)].reshape(N, H, W, 1)
        valid_H = np.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = np.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = np.meshgrid(
            np.linspace(0, H - 1, H, dtype=np.float32),
            np.linspace(0, W - 1, W, dtype=np.float32),
        )
        
        # there is a transpose between torch and np meshgrid
        grid_y = grid_y.T
        grid_x = grid_x.T
        
        grid = np.concatenate([grid_x[..., np.newaxis], grid_y[..., np.newaxis]], axis=-1)

        scale = np.concatenate([valid_W[..., np.newaxis], valid_H[..., np.newaxis]], axis=1).reshape(N, 1, 1, 2)
        grid = (grid[np.newaxis, ...] + 0.5) / scale

        wh = np.ones_like(grid) * 0.05 * (2.0**lvl)

        proposal = np.concatenate((grid, wh), axis=-1).reshape(N, -1, 4)
        proposals.append(proposal)
        _cur += H * W

    output_proposals = np.concatenate(proposals, axis=1)
    return output_proposals