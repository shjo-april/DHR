import cv2
import cmapy
import torch
import numpy as np

from PIL import ImageFont, ImageDraw, Image

ESC = 27
SPACE = 32

MOUSE_MOVE = cv2.EVENT_MOUSEMOVE
MOUSE_WHEEL = cv2.EVENT_MOUSEWHEEL
MOUSE_LBUTTON = cv2.EVENT_LBUTTONDOWN
MOUSE_RBUTTON = cv2.EVENT_RBUTTONDOWN

rectangle = cv2.rectangle

def resize(image, size=None, scale=None, mode='bicubic'):
    inp_dict = {
        'bicubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST,
    }
    if scale is not None:
        h, w = image.shape[:2]
        size = (int(w * scale), int(h * scale))
    return cv2.resize(image, size, interpolation=inp_dict[mode])

def imread(path, mode='opencv'):
    if mode == 'opencv': return cv2.imread(path)
    else: return Image.open(path)

def imwrite(path, image, palette=None):
    if palette is None: cv2.imwrite(path, image)
    else:
        image = Image.fromarray(image.astype(np.uint8)).convert('P')
        image.putpalette(palette)
        image.save(path)

def imshow(winname, image, wait=-1, title=''):
    cv2.imshow(winname, image)

    if len(title) > 0:
        cv2.setWindowTitle(winname, title)

    key = None
    if wait >= 0: 
        key = cv2.waitKey(wait)

    return key

def set_mouse_events(winname, func):
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname, func)

def get_colors(num_classes=20, ignore_index=255, color_format='RGB', data=None):
    if data == 'COCO-Stuff': num_classes += 1

    colors = []
    bitget = lambda v, i: (v & (1 << i)) != 0

    for i in range(num_classes):
        r = g = b = 0
        c = i

        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        colors.append([r, g, b])
    
    while len(colors) < ignore_index: colors.append([0, 0, 0])
    colors.append([224, 224, 192])

    colors = np.asarray(colors, dtype=np.uint8)

    if data is not None:
        if data == 'ADE2016':
            updated_colors = [
                # [0, 0, 0],
                [120, 120, 120],
                [180, 120, 120],
                [6, 230, 230],
                [80, 50, 50],
                [4, 200, 3],
                [120, 120, 80],
                [140, 140, 140],
                [204, 5, 255],
                [230, 230, 230],
                [4, 250, 7],
                [224, 5, 255],
                [235, 255, 7],
                [150, 5, 61],
                [120, 120, 70],
                [8, 255, 51],
                [255, 6, 82],
                [143, 255, 140],
                [204, 255, 4],
                [255, 51, 7],
                [204, 70, 3],
                [0, 102, 200],
                [61, 230, 250],
                [255, 6, 51],
                [11, 102, 255],
                [255, 7, 71],
                [255, 9, 224],
                [9, 7, 230],
                [220, 220, 220],
                [255, 9, 92],
                [112, 9, 255],
                [8, 255, 214],
                [7, 255, 224],
                [255, 184, 6],
                [10, 255, 71],
                [255, 41, 10],
                [7, 255, 255],
                [224, 255, 8],
                [102, 8, 255],
                [255, 61, 6],
                [255, 194, 7],
                [255, 122, 8],
                [0, 255, 20],
                [255, 8, 41],
                [255, 5, 153],
                [6, 51, 255],
                [235, 12, 255],
                [160, 150, 20],
                [0, 163, 255],
                [140, 140, 140],
                [250, 10, 15],
                [20, 255, 0],
                [31, 255, 0],
                [255, 31, 0],
                [255, 224, 0],
                [153, 255, 0],
                [0, 0, 255],
                [255, 71, 0],
                [0, 235, 255],
                [0, 173, 255],
                [31, 0, 255],
                [11, 200, 200],
                [255, 82, 0],
                [0, 255, 245],
                [0, 61, 255],
                [0, 255, 112],
                [0, 255, 133],
                [255, 0, 0],
                [255, 163, 0],
                [255, 102, 0],
                [194, 255, 0],
                [0, 143, 255],
                [51, 255, 0],
                [0, 82, 255],
                [0, 255, 41],
                [0, 255, 173],
                [10, 0, 255],
                [173, 255, 0],
                [0, 255, 153],
                [255, 92, 0],
                [255, 0, 255],
                [255, 0, 245],
                [255, 0, 102],
                [255, 173, 0],
                [255, 0, 20],
                [255, 184, 184],
                [0, 31, 255],
                [0, 255, 61],
                [0, 71, 255],
                [255, 0, 204],
                [0, 255, 194],
                [0, 255, 82],
                [0, 10, 255],
                [0, 112, 255],
                [51, 0, 255],
                [0, 194, 255],
                [0, 122, 255],
                [0, 255, 163],
                [255, 153, 0],
                [0, 255, 10],
                [255, 112, 0],
                [143, 255, 0],
                [82, 0, 255],
                [163, 255, 0],
                [255, 235, 0],
                [8, 184, 170],
                [133, 0, 255],
                [0, 255, 92],
                [184, 0, 255],
                [255, 0, 31],
                [0, 184, 255],
                [0, 214, 255],
                [255, 0, 112],
                [92, 255, 0],
                [0, 224, 255],
                [112, 224, 255],
                [70, 184, 160],
                [163, 0, 255],
                [153, 0, 255],
                [71, 255, 0],
                [255, 0, 163],
                [255, 204, 0],
                [255, 0, 143],
                [0, 255, 235],
                [133, 255, 0],
                [255, 0, 235],
                [245, 0, 255],
                [255, 0, 122],
                [255, 245, 0],
                [10, 190, 212],
                [214, 255, 0],
                [0, 204, 255],
                [20, 0, 255],
                [255, 255, 0],
                [0, 153, 255],
                [0, 41, 255],
                [0, 255, 204],
                [41, 0, 255],
                [41, 255, 0],
                [173, 0, 255],
                [0, 245, 255],
                [71, 0, 255],
                [122, 0, 255],
                [0, 255, 184],
                [0, 92, 255],
                [184, 255, 0],
                [0, 133, 255],
                [255, 214, 0],
                [25, 194, 194],
                [102, 255, 0],
                [92, 0, 255],
            ]
            for i, color in enumerate(updated_colors): colors[i] = color
        elif data == 'Cityscapes':
            updated_colors = [
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [70, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
            ]
            for i, color in enumerate(updated_colors): colors[i] = color
        elif data == 'COCO-Stuff':
            updated_colors = colors.copy()
            for i in range(num_classes-1):
                updated_colors[i] = updated_colors[i+1]
            for i, color in enumerate(updated_colors): colors[i] = color
        elif data == 'Mapillary-Vistas':
            updated_colors = np.asarray([
                [165, 42, 42],
                [0, 192, 0],
                [196, 196, 196],
                [190, 153, 153],
                [180, 165, 180],
                [102, 102, 156],
                [102, 102, 156],
                [128, 64, 255],
                [140, 140, 200],
                [170, 170, 170],
                [250, 170, 160],
                [96, 96, 96],
                [230, 150, 140],
                [128, 64, 128],
                [110, 110, 110],
                [244, 35, 232],
                [150, 100, 100],
                [70, 70, 70],
                [150, 120, 90],
                [220, 20, 60],
                [255, 0, 0],
                [255, 0, 0],
                [255, 0, 0],
                [200, 128, 128],
                [255, 255, 255],
                [64, 170, 64],
                [128, 64, 64],
                [70, 130, 180],
                [255, 255, 255],
                [152, 251, 152],
                [107, 142, 35],
                [0, 170, 30],
                [255, 255, 128],
                [250, 0, 30],
                [0, 0, 0],
                [220, 220, 220],
                [170, 170, 170],
                [222, 40, 40],
                [100, 170, 30],
                [40, 40, 40],
                [33, 33, 33],
                [170, 170, 170],
                [0, 0, 142],
                [170, 170, 170],
                [210, 170, 100],
                [153, 153, 153],
                [128, 128, 128],
                [0, 0, 142],
                [250, 170, 30],
                [192, 192, 192],
                [220, 220, 0],
                [180, 165, 180],
                [119, 11, 32],
                [0, 0, 142],
                [0, 60, 100],
                [0, 0, 142],
                [0, 0, 90],
                [0, 0, 230],
                [0, 80, 100],
                [128, 64, 64],
                [0, 0, 110],
                [0, 0, 70],
                [0, 0, 192],
                [32, 32, 32],
                [0, 0, 0],
                [0, 0, 0],
                ])
            for i, color in enumerate(updated_colors): colors[i] = color
        elif data == 'PascalContext':
            updated_colors = get_colors(num_classes+1)
            updated_colors[:num_classes] = updated_colors[1:num_classes+1]
            for i, color in enumerate(updated_colors): colors[i] = color
        else:
            updated_colors = None
            # raise ValueError(f'Please check {data}')

    if color_format == 'BGR': colors = colors[:, ::-1] # RGB to BGR
    return colors

def apply_colormap(cam, option='SEISMIC'):
    color_dict = {
        'JET': cv2.COLORMAP_JET,
        'HOT': cv2.COLORMAP_HOT,
        'SUMMER': cv2.COLORMAP_SUMMER,
        'WINTER': cv2.COLORMAP_WINTER,
        'GRAY': cmapy.cmap('gray'),
        'SEISMIC': cmapy.cmap('seismic'),
    }
    
    if cam.dtype in [np.float32, np.float64]:
        cam = (cam * 255).astype(np.uint8)
    
    if len(cam.shape) == 3:
        cam = np.max(cam, axis=0)
    
    if option in color_dict:
        cam = cv2.applyColorMap(cam, color_dict[option])
    
    return cam

def draw_text(image: np.ndarray, text: str, coordinate: tuple, color: tuple=(0, 0, 0), font_path: str='./weights/Times New Roman MT Std.otf', font_size: int=20, background: tuple=(79, 244, 255), centering: bool=True, padding: int=5):
    text = ' ' + text
    font = ImageFont.truetype(font_path, font_size)
    
    tw, th = font.getsize(text)
    if centering:
        coordinate = list(coordinate)
        coordinate[0] = max(coordinate[0] - (tw // 2 + padding // 2), 0)
        coordinate[1] = max(coordinate[1] - (th // 2 + padding // 2), 0)
        coordinate = tuple(coordinate)
    
    if background is not None:
        cv2.rectangle(image, coordinate, (coordinate[0] + tw + padding, coordinate[1] + th + padding), background, cv2.FILLED)

    pillow_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pillow_image)
    draw.text(coordinate, text, font=font, fill=(color[0], color[1], color[2], 0))

    image[:, :, :] = np.asarray(pillow_image)

def draw_circle(image, point, size, color, edge_color=(0, 0, 0)):
    x, y = point
    edge_size = max(size // 5, 1)

    b, g, r = color
    eb, eg, er = edge_color

    pillow_image = Image.fromarray(image)
    drw = ImageDraw.Draw(pillow_image)
    drw.ellipse(
        [(x-size, y-size), (x+size, y+size)], 
        (b, g, r, 0), (eb, eg, er, 0), edge_size
    )
    image[:, :, :] = np.asarray(pillow_image)

def visualize_heatmaps(heatmaps, tags=None, image=None, option='SEISMIC', norm=False, image_tag='Image', font_size=40, bg_colors=None):
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()
    
    vis_heatmaps = []
    
    # if image is not None:
    #     draw_text(image, image_tag, (0, 0), font_size=font_size)
    #     vis_heatmaps.append(image)

    if tags is None:
        tags = [None for _ in heatmaps]

    if bg_colors is None:
        bg_colors = [(79, 244, 255) for _ in heatmaps]

    for tag, heatmap, bg_color in zip(tags, heatmaps, bg_colors):
        if norm: 
            min_v = heatmap.min()
            max_v = heatmap.max()
            heatmap = (heatmap - min_v) / (max_v - min_v + 1e-5)
        
        heatmap = apply_colormap(heatmap, option)
        if image is not None: heatmap = cv2.addWeighted(image, 0.4, heatmap, 0.6, 0.0)
        if tag is not None: draw_text(heatmap, tag, (0, 0), font_size=font_size, background=bg_color)
        vis_heatmaps.append(heatmap)

    # print(len(vis_heatmaps))

    return np.concatenate(vis_heatmaps, axis=1)