"""
Hero GIF (CPU only): compose figures/hero.gif from figures of the paper.

Four scenes: the vanishing problem of adjacent minor classes (paper Fig. 2), the complementary USS and WSS
features (Fig. 3 and Fig. 4), the mask of two training images after each step of DHR (Fig. 5 and Table 6),
and the results on five benchmarks (Fig. 9, Table 2 and Table 4). Every image and number comes from
hero_gif/panels, hero_gif/logos and hero_gif/scenes.py, so no model and no dataset are needed.

    python hero_gif/build_gif.py                  # writes figures/hero.gif
    python hero_gif/build_gif.py --preview DIR    # writes every frame to DIR as a PNG instead

The committed GIF was rendered on Windows with Segoe UI. On other systems the script falls back to the
DejaVu Sans bundled with matplotlib: same layout, slightly different glyphs.
"""
import argparse
import json
import os
import runpy

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PANELS = os.path.join(HERE, 'panels')

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(ROOT, 'figures', 'hero.gif'))
ap.add_argument('--preview', metavar='DIR', help='write the frames as PNGs to DIR and skip the GIF')
args = ap.parse_args()

W, H = 1280, 960
TEAL, TEAL_BG, MINT_LINE = (0, 102, 102), (232, 245, 244), (150, 205, 200)
INK, MUTED, LINE, WHITE, GREY = (16, 24, 31), (91, 105, 117), (220, 229, 233), (255, 255, 255), (178, 188, 196)
TAB_DIM = (191, 217, 217)                                          # inactive scene tab
BASE = (0, 110, 175)                                               # other methods and the input mask
SOFT = (215, 238, 236)                                             # small text on a solid teal card
RED, GREEN = (214, 40, 40), (16, 160, 72)                          # boxes: minor class missing / restored
M, TOP, NOTE_Y = 24, 214, 938                                      # side margin · top of the panels · baseline note
V = runpy.run_path(os.path.join(HERE, 'scenes.py'))['SCENES']

WIN_FONTS = 'C:/Windows/Fonts'
FALLBACK = {'segoeui.ttf': 'DejaVuSans.ttf', 'seguisb.ttf': 'DejaVuSans-Bold.ttf', 'segoeuib.ttf': 'DejaVuSans-Bold.ttf'}


def F(name, size):
    path = os.path.join(WIN_FONTS, name)
    if not os.path.exists(path):
        import matplotlib
        path = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', FALLBACK[name])
    return ImageFont.truetype(path, size)


F_TITLE, F_BADGE, F_SUBT, F_TAB = F('segoeuib.ttf', 40), F('seguisb.ttf', 16), F('segoeui.ttf', 17), F('segoeuib.ttf', 16)
F_HEAD, F_H, F_HB = F('segoeuib.ttf', 22), F('segoeuib.ttf', 20), F('segoeuib.ttf', 17)
F_MSG, F_NOTE, F_TICK = F('segoeuib.ttf', 20), F('segoeui.ttf', 15), F('segoeui.ttf', 14)
F_NUM, F_BIG, F_STEP, F_CHIP, F_AFF = F('segoeuib.ttf', 32), F('segoeuib.ttf', 40), F('segoeuib.ttf', 13), F('seguisb.ttf', 14), F('seguisb.ttf', 17)


def logo(name, height):
    im = Image.open(os.path.join(HERE, 'logos', name)).convert('RGBA')
    return im.resize((round(im.width * height / im.height), height), Image.LANCZOS)


# the affiliations of the paper, in paper order: OGQ, University of Michigan | Samsung Electronics, MGH and Harvard Medical School
LOGOS_L = [logo('ogq.png', 40), logo('umich.png', 34)]
LOGOS_R = [logo('samsung.png', 22), logo('mgh.png', 46), logo('hms.png', 46)]
LOGO_Y = 42                                                        # vertical centre of the logo row


def load(rel, size):
    """A panel at its display size. The build stops if that would change the aspect ratio."""
    im = Image.open(os.path.join(PANELS, rel)).convert('RGB')
    assert abs(im.width / im.height - size[0] / size[1]) < 0.02, (rel, im.size, size)
    return im if im.size == size else im.resize(size, Image.LANCZOS)


def rrect(dr, box, r, fill=None, outline=None, width=1):
    dr.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def wrap(dr, text, font, width):
    lines, cur = [], ''
    for word in text.split():
        t = (cur + ' ' + word).strip()
        if cur and dr.textlength(t, font=font) > width:
            lines.append(cur); cur = word
        else:
            cur = t
    return lines + [cur]


def line(dr, y, text, font, fill):
    """One centered line; the build stops if it does not fit between the side margins."""
    assert dr.textlength(text, font=font) <= W - 2 * M, text
    dr.text((W / 2, y), text, font=font, fill=fill, anchor='mm')


def text_in(dr, xy, text, font, fill, width, anchor='lm'):
    """One line that must fit into the given width."""
    assert dr.textlength(text, font=font) <= width, text
    dr.text(xy, text, font=font, fill=fill, anchor=anchor)


def canvas(scene, head, note):
    """White frame with the header: logos, title and venue, subtitle, scene tabs, the scene headline and its note."""
    im = Image.new('RGB', (W, H), WHITE); dr = ImageDraw.Draw(im)
    tw = dr.textlength(V['title'], font=F_TITLE); bw = dr.textlength(V['badge'], font=F_BADGE) + 28
    x0 = (W - (tw + 14 + bw)) / 2
    x = 34
    for lg in LOGOS_L:
        im.paste(lg, (x, LOGO_Y - lg.height // 2), lg); x += lg.width + 26
    assert x - 26 < x0 - 24, 'left logos touch the title'
    x = W - 34
    for k, lg in reversed(list(enumerate(LOGOS_R))):
        x -= lg.width
        im.paste(lg, (x, LOGO_Y - lg.height // 2), lg)
        if k == 1:                                                 # "Samsung Electronics": the wordmark, then the word
            x -= 26
            dr.text((x, LOGO_Y), 'Electronics', font=F_AFF, fill=(58, 72, 83), anchor='rm')
            x -= round(dr.textlength('Electronics', font=F_AFF)) + 8
        else:
            x -= 12
    assert x > x0 + tw + 14 + bw + 24, 'right logos touch the title'
    dr.text((x0, 6), V['title'], font=F_TITLE, fill=TEAL)
    rrect(dr, (x0 + tw + 14, 22, x0 + tw + 14 + bw, 50), 14, fill=TEAL)
    dr.text((x0 + tw + 14 + bw / 2, 36), V['badge'], font=F_BADGE, fill=WHITE, anchor='mm')
    line(dr, 74, V['subtitle'], F_SUBT, MUTED)
    tabs = V['tabs']                                               # fixed position and size, only the current one is solid
    tw_ = (W - 2 * M - 16 * (len(tabs) - 1)) / len(tabs); x = M
    for i, t in enumerate(tabs):
        rrect(dr, (x, 96, x + tw_, 122), 8, fill=TEAL if i == scene else TAB_DIM)
        dr.text((x + tw_ / 2, 109), t, font=F_TAB, fill=WHITE if i == scene else TEAL, anchor='mm')
        x += tw_ + 16
    line(dr, 154, head, F_HEAD, INK)
    line(dr, NOTE_Y, note, F_NOTE, MUTED)
    return im, dr


def framed(im, dr, pic, x, y, col, width=3):
    im.paste(pic, (x, y))
    dr.rectangle((x - 2, y - 2, x + pic.width + 1, y + pic.height + 1), outline=col, width=width)


def placeholder(dr, x, y, w, h, label):
    rrect(dr, (x, y, x + w - 1, y + h - 1), 10, fill=TEAL_BG, outline=MINT_LINE)
    dr.text((x + w / 2, y + h / 2), label, font=F_HEAD, fill=MINT_LINE, anchor='mm')


def chips(dr, x, y, label, tags):
    """A row of class chips in the colours of the masks; y is the vertical centre."""
    dr.text((x, y), label, font=F_CHIP, fill=MUTED, anchor='lm')
    x += dr.textlength(label, font=F_CHIP) + 10
    for name, col in tags:
        col = tuple(col); w = dr.textlength(name, font=F_CHIP) + 20
        rrect(dr, (x, y - 12, x + w, y + 12), 6, fill=col)
        dark = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2] < 150
        dr.text((x + w / 2, y - 1), name, font=F_CHIP, fill=WHITE if dark else INK, anchor='mm')
        x += w + 8
    assert x <= W - M, tags


def tag_colors(*tag_lists):
    return [tuple(c) for tags in tag_lists for _, c in tags]


# ---------------------------------------------------------------- scene 1: the vanishing problem (paper Fig. 2)
PB = V['problem']
PW, PGAP, PROW = 238, 10, 238 + 46
PX = [(W - (5 * PW + 4 * PGAP)) // 2 + k * (PW + PGAP) for k in range(5)]
PROBLEM = [[load('problem/%s/%s' % (r['id'], f), (PW, PW)) for f, _ in PB['columns']] for r in PB['rows']]


def problem_frame(s):
    im, dr = canvas(0, 'Minor classes next to a larger class vanish from WSS masks', PB['note'])
    shown = [True, True, s >= 1, s >= 1, s >= 2]
    for k, (_, title) in enumerate(PB['columns']):
        col = TEAL if k == 4 else BASE if k >= 2 else INK
        dr.text((PX[k] + PW / 2, TOP - 10), title, font=F_H, fill=col if shown[k] else GREY, anchor='ms')
        for r, row in enumerate(PB['rows']):
            y = TOP + r * PROW
            if not shown[k]:
                placeholder(dr, PX[k], y, PW, PW, title)
                continue
            framed(im, dr, PROBLEM[r][k], PX[k], y, col if k >= 2 else LINE)
            if k >= 2:                                             # the minor class: missing (red) or restored (green)
                x0, y0, x1, y1 = (round(v * PW) for v in row['box'])
                dr.rectangle((PX[k] + x0, y + y0, PX[k] + x1 - 1, y + y1 - 1), outline=GREEN if k == 4 else RED, width=3)
    for r, row in enumerate(PB['rows']):
        chips(dr, PX[0], TOP + r * PROW + PW + 22, 'Class labels', row['tags'])
    line(dr, 836, PB['messages'][s], F_MSG, TEAL if s == 2 else BASE if s == 1 else INK)
    if s >= 1:
        line(dr, 876, PB['stat'], F_NOTE, MUTED)
    return im, tag_colors(*(r['tags'] for r in PB['rows']))


# ---------------------------------------------------------------- scene 2: USS and WSS features (paper Fig. 3 and Fig. 4)
FT = V['features']
FEAT = [(Image.open(os.path.join(PANELS, p)).convert('RGB'), t, a, b) for p, t, a, b in FT['panels']]
FW, FH = (W - 2 * M - 16) // 2, FEAT[0][0].height + 20              # each half of Fig. 3 sits in a white panel with a 10 px margin
assert all(pic.size == (FW - 20, FH - 20) for pic, *_ in FEAT)
HIER = Image.open(os.path.join(PANELS, FT['hierarchy'])).convert('RGB')


def features_frame(s):
    im, dr = canvas(1, 'USS and WSS features separate different kinds of regions', FT['note'])
    for k, (pic, title, a, b) in enumerate(FEAT):
        x, on = M + k * (FW + 16), k <= s
        dr.text((x, TOP - 10), title, font=F_H, fill=TEAL if on else GREY, anchor='ls')
        if on:
            im.paste(pic, (x + 10, TOP + 10))
            dr.rectangle((x - 2, TOP - 2, x + FW + 1, TOP + FH + 1), outline=TEAL if k == s else LINE, width=3)
        else:
            placeholder(dr, x, TOP, FW, FH, title[:3])
        y = TOP + FH + 22
        rrect(dr, (x, y, x + FW - 1, y + 92), 12, fill=(TEAL if k == s else TEAL_BG) if on else WHITE, outline=None if on and k == s else MINT_LINE if on else LINE)
        text_in(dr, (x + 20, y + 32), a, F_H, (WHITE if k == s else INK) if on else GREY, FW - 40)
        text_in(dr, (x + 20, y + 63), b, F_NOTE, (SOFT if k == s else MUTED) if on else GREY, FW - 40)
    line(dr, 836, FT['messages'][s], F_MSG, TEAL)
    return im, []


def hierarchy_frame():
    im, dr = canvas(1, 'DHR uses both, hierarchically', FT['hierarchy_note'])
    im.paste(HIER, ((W - HIER.width) // 2, 190))
    assert 190 + HIER.height <= 846
    line(dr, 884, FT['hierarchy_message'], F_MSG, TEAL)
    return im, []


# ---------------------------------------------------------------- scene 3: the mask after each step (paper Fig. 5 and Table 6)
MT = V['method']
CW, CH, CGAP, CROW = 299, 224, 12, 224 + 48
CX = [M + k * (CW + CGAP) for k in range(4)]
STEPS = MT['steps']
METHOD = [{f: load('method/%s/%s' % (r['id'], f), (CW, CH)) for f in ['image.jpg', 'gt.png'] + [st[0] for st in STEPS]} for r in MT['rows']]
CARD_Y = TOP + 2 * CROW


def method_frame(s, final=False):
    im, dr = canvas(2, 'DHR rebalances the input WSS mask in three steps', MT['note'])
    titles = [('Image', INK), ('Input WSS mask', BASE), ('After Step %d' % max(s, 1), TEAL if s else GREY), ('Ground truth', INK)]
    for k, (title, col) in enumerate(titles):
        dr.text((CX[k] + CW / 2, TOP - 10), title, font=F_H, fill=col, anchor='ms')
    for r, row in enumerate(MT['rows']):
        y = TOP + r * CROW
        framed(im, dr, METHOD[r]['image.jpg'], CX[0], y, LINE)
        framed(im, dr, METHOD[r]['input.png'], CX[1], y, BASE)
        if s:
            framed(im, dr, METHOD[r][STEPS[s][0]], CX[2], y, TEAL)
        else:
            placeholder(dr, CX[2], y, CW, CH, 'Step 1')
        framed(im, dr, METHOD[r]['gt.png'], CX[3], y, LINE)
        chips(dr, CX[0], y + CH + 23, 'Class labels', row['tags'])
    for k, (_, label, title, sub, value, _) in enumerate(STEPS):
        x, y = CX[k], CARD_Y
        on, cur = k <= s, k == s and not final
        fg = WHITE if cur else INK if on else GREY
        rrect(dr, (x, y, x + CW - 1, y + 98), 12, fill=TEAL if cur else TEAL_BG if on else WHITE, outline=None if cur else MINT_LINE if on else LINE)
        dr.text((x + 16, y + 24), label, font=F_STEP, fill=SOFT if cur else TEAL if on else GREY, anchor='lm')
        text_in(dr, (x + 16, y + 52), title, F_HB, fg, CW - 32)
        text_in(dr, (x + 16, y + 77), sub, F_TICK, SOFT if cur else MUTED if on else GREY, CW - 32)
        if on:
            dr.text((x + CW - 16, y + 26), '%.1f' % value, font=F_NUM, fill=WHITE if cur else TEAL, anchor='rm')
            if k:
                dr.text((x + CW - 16 - dr.textlength('%.1f' % value, font=F_NUM) - 10, y + 29), '%+.1f' % (value - STEPS[0][4]), font=F_TICK, fill=SOFT if cur else MUTED, anchor='rm')
    line(dr, 892, MT['summary'] if final else STEPS[s][5], F_MSG, TEAL if s else BASE)
    return im, tag_colors(*(r['tags'] for r in MT['rows']))


# ---------------------------------------------------------------- scene 4: results on five benchmarks (paper Fig. 9, Table 2, Table 4)
RS = V['results']
BW, SW, RGAP = 300, 222, 12                                        # 3 large and 4 small panels share one width: 3 * 300 + 2 * 12 = 4 * 222 + 3 * 12
SMALL_Y = TOP + BW + 38
RX0, RX1 = M + 3 * BW + 2 * RGAP + 24, W - M                       # right column: the numbers
RESULTS = {d['key']: {f: load('results/%s/%s' % (d['key'], f), (BW, BW) if k < 3 else (SW, SW))
                      for k, (f, _) in enumerate(RS['big'] + RS['small'])} for d in RS['datasets']}


def results_frame(n):
    D = RS['datasets'][n]
    im, dr = canvas(3, 'DHR on five benchmarks, with image-level class labels only', RS['note'])
    for k, (f, title) in enumerate(RS['big']):
        x = M + k * (BW + RGAP)
        col = TEAL if f == 'dhr.png' else INK
        dr.text((x + BW / 2, TOP - 10), title + (' · ' + D['name'] if k == 0 else ''), font=F_H, fill=col, anchor='ms')
        framed(im, dr, RESULTS[D['key']][f], x, TOP, TEAL if f == 'dhr.png' else LINE)
    for k, (f, title) in enumerate(RS['small']):
        x = M + k * (SW + RGAP)
        dr.text((x + SW / 2, SMALL_Y - 10), title, font=F_H, fill=BASE, anchor='ms')
        framed(im, dr, RESULTS[D['key']][f], x, SMALL_Y, BASE)
    chips(dr, M, SMALL_Y + SW + 26, 'Tags', D['tags'])
    dr.text((RX0, TOP - 10), 'mIoU (%), ResNet-101', font=F_H, fill=INK, anchor='ls')
    for k, d in enumerate(RS['datasets']):
        y, cur = TOP + k * 86, k == n
        rrect(dr, (RX0, y, RX1, y + 76), 12, fill=TEAL if cur else TEAL_BG, outline=None if cur else MINT_LINE)
        text_in(dr, (RX0 + 16, y + 26), '%s · %s' % (d['name'], d['split']), F_HB, WHITE if cur else INK, RX1 - RX0 - 110)
        dr.text((RX0 + 16, y + 52), 'MARS %.1f' % d['mars'], font=F_TICK, fill=SOFT if cur else MUTED, anchor='lm')
        dr.text((RX1 - 16, y + 38), '%.1f' % d['dhr'], font=F_NUM, fill=WHITE if cur else TEAL, anchor='rm')
    y = TOP + 5 * 86
    rrect(dr, (RX0, y, RX1, SMALL_Y + SW + 38), 12, fill=WHITE, outline=MINT_LINE)
    dr.text((RX0 + 16, y + 34), RS['gap'][0], font=F_BIG, fill=TEAL, anchor='lm')
    for i, t in enumerate(wrap(dr, RS['gap'][1], F_TICK, RX1 - RX0 - 32)):
        dr.text((RX0 + 16, y + 70 + i * 19), t, font=F_TICK, fill=MUTED, anchor='lm')
        assert y + 70 + i * 19 + 10 < SMALL_Y + SW + 38, 'the gap card is too small for its text'
    line(dr, 880, RS['message'], F_MSG, TEAL)
    return im, tag_colors(D['tags'])


frames = []
add = lambda fr, ms: frames.append((fr[0], ms, fr[1]))
for s, ms in enumerate((1800, 2600, 3200)):
    add(problem_frame(s), ms)
for s, ms in enumerate((2600, 2800)):
    add(features_frame(s), ms)
add(hierarchy_frame(), 3800)
for s, ms in enumerate((1800, 2200, 2200, 2400)):
    add(method_frame(s), ms)
add(method_frame(3, final=True), 3000)
for n, ms in enumerate((2200, 1800, 1800, 1800, 3400)):
    add(results_frame(n), ms)

# GIF palette: the flat UI colors, the main logo colors and the class colors of the chips are reserved so text, logos and chips stay clean;
# the rest of each frame is quantized.
def logo_colors(im, n=7):
    flat = Image.new('RGB', im.size, WHITE); flat.paste(im, (0, 0), im)
    p = flat.quantize(colors=n, method=Image.MEDIANCUT).getpalette()[:3 * n]
    return [tuple(p[i:i + 3]) for i in range(0, len(p), 3)]


RESERVED = list(dict.fromkeys([TEAL, TEAL_BG, MINT_LINE, INK, MUTED, LINE, WHITE, (0, 0, 0), GREY, TAB_DIM, BASE, SOFT, RED, GREEN, (58, 72, 83)]
                              + [c for lg in LOGOS_L + LOGOS_R for c in logo_colors(lg)]))


def to_palette(f, extra):
    fixed_colors = list(dict.fromkeys(RESERVED + extra))
    k = len(fixed_colors)
    q = f.quantize(colors=256 - k, method=Image.MEDIANCUT)
    p = (q.getpalette() or [])[:3 * (256 - k)]
    p = [v for c in fixed_colors for v in c] + p + [0] * (3 * (256 - k) - len(p))
    pi = Image.new('P', (1, 1)); pi.putpalette(p)
    out = f.quantize(palette=pi, dither=Image.FLOYDSTEINBERG)
    idx, rgb = np.array(out), np.array(f.convert('RGB'))
    for j, col in enumerate(fixed_colors):
        idx[(rgb == col).all(-1)] = j
    fixed = Image.frombytes('P', out.size, idx.astype(np.uint8).tobytes()); fixed.putpalette(p)
    return fixed


total_ms = sum(ms for _, ms, _ in frames)
if args.preview:
    os.makedirs(args.preview, exist_ok=True)
    for i, (f, _, _) in enumerate(frames):
        f.save(os.path.join(args.preview, f'frame_{i:03d}.png'))
    json.dump([ms for _, ms, _ in frames], open(os.path.join(args.preview, 'durations.json'), 'w'))
    print(f'{len(frames)} frames ({total_ms / 1000:.1f} s) written to {args.preview}')
else:
    before = open(args.out, 'rb').read() if os.path.exists(args.out) else None
    pal = [to_palette(f, extra) for f, _, extra in frames]
    pal[0].save(args.out, save_all=True, append_images=pal[1:], duration=[ms for _, ms, _ in frames], loop=0, optimize=True, disposal=1)
    after = open(args.out, 'rb').read()
    state = 'new file' if before is None else 'identical to the previous file' if before == after else 'differs from the previous file'
    print(f'{args.out}: {len(frames)} frames, {total_ms / 1000:.1f} s, {len(after) / 1e6:.1f} MB ({state})')
