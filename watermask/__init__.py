import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import copy

from watermask.blendmodes import BLEND_MODES


def image_blend_advance_v2(background_image, layer_image,
                           invert_mask, blend_mode, opacity,
                           x_percent, y_percent,
                           mirror, scale, aspect_ratio, rotate,
                           transform_method, anti_aliasing,
                           layer_mask=None
                           ):
    b_images = []
    l_images = []
    l_masks = []
    ret_images = []
    ret_masks = []
    for b in background_image:
        b_images.append(torch.unsqueeze(b, 0))
    for l in layer_image:
        l_images.append(torch.unsqueeze(l, 0))
        m = tensor2pil(l)
        if m.mode == 'RGBA':
            l_masks.append(m.split()[-1])
        else:
            l_masks.append(Image.new('L', m.size, 'white'))
    if layer_mask is not None:
        if layer_mask.dim() == 2:
            layer_mask = torch.unsqueeze(layer_mask, 0)
        l_masks = []
        for m in layer_mask:
            if invert_mask:
                m = 1 - m
            l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

    max_batch = max(len(b_images), len(l_images), len(l_masks))
    for i in range(max_batch):
        background_image = b_images[i] if i < len(b_images) else b_images[-1]
        layer_image = l_images[i] if i < len(l_images) else l_images[-1]
        _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
        # preprocess
        _canvas = tensor2pil(background_image).convert('RGB')
        _layer = tensor2pil(layer_image)

        if _mask.size != _layer.size:
            _mask = Image.new('L', _layer.size, 'white')
            # log(f"Warning: {NODE_NAME} mask mismatch, dropped!", message_type='warning')

        orig_layer_width = _layer.width
        orig_layer_height = _layer.height
        _mask = _mask.convert("RGB")

        target_layer_width = int(orig_layer_width * scale)
        target_layer_height = int(orig_layer_height * scale * aspect_ratio)

        # mirror
        if mirror == 'horizontal':
            _layer = _layer.transpose(Image.FLIP_LEFT_RIGHT)
            _mask = _mask.transpose(Image.FLIP_LEFT_RIGHT)
        elif mirror == 'vertical':
            _layer = _layer.transpose(Image.FLIP_TOP_BOTTOM)
            _mask = _mask.transpose(Image.FLIP_TOP_BOTTOM)

        # scale
        _layer = _layer.resize((target_layer_width, target_layer_height))
        _mask = _mask.resize((target_layer_width, target_layer_height))
        # rotate
        _layer, _mask, _ = image_rotate_extend_with_alpha(_layer, rotate, _mask, transform_method, anti_aliasing)

        # 处理位置
        x = int(_canvas.width * x_percent / 100 - _layer.width / 2)
        y = int(_canvas.height * y_percent / 100 - _layer.height / 2)

        # composit layer
        _comp = copy.copy(_canvas)
        _compmask = Image.new("RGB", _comp.size, color='black')
        _comp.paste(_layer, (x, y))
        _compmask.paste(_mask, (x, y))
        _compmask = _compmask.convert('L')
        _comp = chop_image_v2(_canvas, _comp, blend_mode, opacity)

        # composition background
        _canvas.paste(_comp, mask=_compmask)

        ret_images.append(pil2tensor(_canvas))
        ret_masks.append(image2mask(_compmask))

    # log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
    return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

def image2mask(image:Image) -> torch.Tensor:
    if image.mode == 'L':
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        image = image.convert('RGB').split()[0]
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def image_rotate_extend_with_alpha(image:Image, angle:float, alpha:Image=None, method:str="lanczos", SSAA:int=0) -> tuple:
    _image = __rotate_expand(image.convert('RGB'), angle, SSAA, method)
    if angle is not None:
        _alpha = __rotate_expand(alpha.convert('RGB'), angle, SSAA, method)
        ret_image = RGB2RGBA(_image, _alpha)
    else:
        ret_image = _image
    return (_image, _alpha.convert('L'), ret_image)

def RGB2RGBA(image:Image, mask:Image) -> Image:
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))

def __rotate_expand(image:Image, angle:float, SSAA:int=0, method:str="lanczos") -> Image:
    images = pil2tensor(image)
    expand = "true"
    height, width = images[0, :, :, 0].shape

    def rotate_tensor(tensor):
        resize_sampler = Image.LANCZOS
        rotate_sampler = Image.BICUBIC
        if method == "bicubic":
            resize_sampler = Image.BICUBIC
            rotate_sampler = Image.BICUBIC
        elif method == "hamming":
            resize_sampler = Image.HAMMING
            rotate_sampler = Image.BILINEAR
        elif method == "bilinear":
            resize_sampler = Image.BILINEAR
            rotate_sampler = Image.BILINEAR
        elif method == "box":
            resize_sampler = Image.BOX
            rotate_sampler = Image.NEAREST
        elif method == "nearest":
            resize_sampler = Image.NEAREST
            rotate_sampler = Image.NEAREST
        img = tensor2pil(tensor)
        if SSAA > 1:
            img_us_scaled = img.resize((width * SSAA, height * SSAA), resize_sampler)
            img_rotated = img_us_scaled.rotate(angle, rotate_sampler, expand == "true", fillcolor=(0, 0, 0, 0))
            img_down_scaled = img_rotated.resize((img_rotated.width // SSAA, img_rotated.height // SSAA), resize_sampler)
            result = pil2tensor(img_down_scaled)
        else:
            img_rotated = img.rotate(angle, rotate_sampler, expand == "true", fillcolor=(0, 0, 0, 0))
            result = pil2tensor(img_rotated)
        return result

    if angle == 0.0 or angle == 360.0:
        return tensor2pil(images)
    else:
        rotated_tensor = torch.stack([rotate_tensor(images[i]) for i in range(len(images))])
        return tensor2pil(rotated_tensor).convert('RGB')

def chop_image_v2(background_image:Image, layer_image:Image, blend_mode:str, opacity:int) -> Image:

    backdrop_prepped = np.asfarray(background_image.convert('RGBA'))
    source_prepped = np.asfarray(layer_image.convert('RGBA'))
    blended_np = BLEND_MODES[blend_mode](backdrop_prepped, source_prepped, opacity / 100)

    # final_tensor = (torch.from_numpy(blended_np / 255)).unsqueeze(0)
    # return tensor2pil(_tensor)

    return Image.fromarray(np.uint8(blended_np)).convert('RGB')

def get_file_extension(file_path):
    # 使用os.path.splitext()分离文件名和扩展名
    _, file_extension = os.path.splitext(file_path)
    return file_extension


if __name__ == '__main__':
    folder = './images'
    logo_path = './logo.png'
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    for filename in os.listdir(folder):
        if filename.lower().endswith(extensions):
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    image,mask = image_blend_advance_v2(pil2tensor(img),pil2tensor(Image.open(logo_path)),False,"normal",
                                                        100,90.00,8.00,None,0.15,1.00,0.00,"lanczos",0,None)
                    add_water_mask_image = tensor2pil(image)
                    add_water_mask_image.save('./results/'+filename)
                    print(f"成功处理图片: {filename}")
            except Exception as e:
                print(f"无法打开图片 {filename}: {e}")
