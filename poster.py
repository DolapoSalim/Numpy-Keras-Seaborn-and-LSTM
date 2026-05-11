import math
import os
from reportlab.lib.pagesizes import A3
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register fonts
pdfmetrics.registerFont(TTFont('Poppins', r'C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\Numpy-Keras-Seaborn-and-LSTM-1\Poppins-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Poppins-Light', r'C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\Numpy-Keras-Seaborn-and-LSTM-1\Poppins-Light.ttf'))
pdfmetrics.registerFont(TTFont('Poppins-Bold', r'C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\Numpy-Keras-Seaborn-and-LSTM-1\Poppins-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Lora', r'C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\Numpy-Keras-Seaborn-and-LSTM-1\Lora-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Lora-Italic', r'C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\Numpy-Keras-Seaborn-and-LSTM-1\Lora-Italic.ttf'))
pdfmetrics.registerFont(TTFont('Baskerville', r"C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\Numpy-Keras-Seaborn-and-LSTM-1\LibreBaskerville-Regular.ttf"))

# --- PALETTE (deep ocean + bioluminescent signals) ---
BG        = HexColor('#0B1320')   # abyssal navy
DEEP      = HexColor('#0D1B2A')   # depth layer
MID       = HexColor('#112236')   # mid-water
SURFACE   = HexColor('#163152')   # surface horizon
TEAL_DK   = HexColor('#1A6B72')   # deep teal
TEAL      = HexColor('#2A9D8F')   # primary teal / CV
TEAL_LT   = HexColor('#52C5B8')   # light teal
CERULEAN  = HexColor('#264B7A')   # remote sensing blue
CERULEAN_LT = HexColor('#4A90C4') # RS highlight
SAND      = HexColor('#C8B98A')   # calcium carbonate / 3D
SAND_LT   = HexColor('#E8D9B0')   # highlight sand
GOLD      = HexColor('#E9C46A')   # EOV signal / accent
AMBER     = HexColor('#F4A261')   # warm accent
CREAM     = HexColor('#F0EAD6')   # label text
WHITE_DIM = HexColor('#A8BDD0')   # secondary text
GRID_LINE = HexColor('#1E3550')   # subtle grid

W, H = A3
cx, cy = W/2, H/2

OUT = r"C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\Numpy-Keras-Seaborn-and-LSTM-1\datasets\poster.pdf"

c = canvas.Canvas(OUT, pagesize=A3)

# ─── BACKGROUND ───────────────────────────────────────────────────────────────
c.setFillColor(BG)
c.rect(0, 0, W, H, fill=1, stroke=0)

# Subtle depth gradient bands
for i, (col, y_frac, h_frac) in enumerate([
    (DEEP, 0.0, 0.25),
    (MID, 0.25, 0.25),
    (SURFACE, 0.50, 0.25),
]):
    c.setFillColor(col)
    c.setFillAlpha(0.18)
    c.rect(0, y_frac * H, W, h_frac * H, fill=1, stroke=0)

c.setFillAlpha(1.0)

# ─── BACKGROUND GRID (bathymetric contours) ───────────────────────────────────
c.setStrokeColor(GRID_LINE)
c.setStrokeAlpha(0.5)
c.setLineWidth(0.3)
step = 18*mm
x = 0
while x <= W:
    c.line(x, 0, x, H)
    x += step
y = 0
while y <= H:
    c.line(0, y, W, y)
    y += step
c.setStrokeAlpha(1.0)

# ─── FAINT CONCENTRIC RINGS (sonar / scale marker) centered on figure ─────────
cx_fig = W * 0.50
cy_fig = H * 0.50
for r in range(1, 18):
    radius = r * 22 * mm
    alpha = max(0.04, 0.14 - r * 0.007)
    c.setStrokeColor(TEAL_LT)
    c.setStrokeAlpha(alpha)
    c.setLineWidth(0.4)
    c.circle(cx_fig, cy_fig, radius, fill=0, stroke=1)
c.setStrokeAlpha(1.0)

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def draw_node(cx, cy, r_outer, r_inner, fill_col, stroke_col, label, sublabel=None, label_col=None):
    """Draw a ringed node with label."""
    c.setFillColor(BG)
    c.setStrokeColor(stroke_col)
    c.setLineWidth(1.2)
    c.circle(cx, cy, r_outer, fill=1, stroke=1)
    c.setFillColor(fill_col)
    c.setStrokeAlpha(0.0)
    c.circle(cx, cy, r_inner, fill=1, stroke=0)
    c.setStrokeAlpha(1.0)
    lc = label_col if label_col else CREAM
    c.setFillColor(lc)
    c.setFont('Poppins-Light', 7)
    tw = c.stringWidth(label, 'Poppins-Light', 7)
    c.drawString(cx - tw/2, cy - 3.5, label)
    if sublabel:
        c.setFillColor(WHITE_DIM)
        c.setFont('Poppins-Light', 5.5)
        tw2 = c.stringWidth(sublabel, 'Poppins-Light', 5.5)
        c.drawString(cx - tw2/2, cy - 10, sublabel)

def draw_dashed_line(x1, y1, x2, y2, col, alpha=0.6, dash=(4,4), lw=0.7):
    c.setStrokeColor(col)
    c.setStrokeAlpha(alpha)
    c.setLineWidth(lw)
    c.setDash(*dash)
    c.line(x1, y1, x2, y2)
    c.setDash()
    c.setStrokeAlpha(1.0)

def draw_solid_line(x1, y1, x2, y2, col, alpha=0.85, lw=1.0):
    c.setStrokeColor(col)
    c.setStrokeAlpha(alpha)
    c.setLineWidth(lw)
    c.line(x1, y1, x2, y2)
    c.setStrokeAlpha(1.0)

def draw_arrow_line(x1, y1, x2, y2, col, alpha=0.7, lw=0.8):
    draw_solid_line(x1, y1, x2, y2, col, alpha, lw)
    # arrowhead
    angle = math.atan2(y2 - y1, x2 - x1)
    ahl = 6
    ahw = 3
    ax1 = x2 - ahl * math.cos(angle) + ahw * math.sin(angle)
    ay1 = y2 - ahl * math.sin(angle) - ahw * math.cos(angle)
    ax2 = x2 - ahl * math.cos(angle) - ahw * math.sin(angle)
    ay2 = y2 - ahl * math.sin(angle) + ahw * math.cos(angle)
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(ax1, ay1)
    p.lineTo(ax2, ay2)
    p.close()
    c.setFillColor(col)
    c.setFillAlpha(alpha)
    c.drawPath(p, fill=1, stroke=0)
    c.setFillAlpha(1.0)

def label_box(bx, by, bw, bh, title, lines, title_col, line_col=None, bg_col=None):
    """Minimal info box."""
    if bg_col:
        c.setFillColor(bg_col)
        c.setFillAlpha(0.18)
        c.roundRect(bx, by, bw, bh, 3*mm, fill=1, stroke=0)
        c.setFillAlpha(1.0)
    c.setFillColor(title_col)
    c.setFont('Poppins-Bold', 6.5)
    c.drawString(bx + 4*mm, by + bh - 9*mm, title)
    lc = line_col if line_col else WHITE_DIM
    c.setFont('Poppins-Light', 5.5)
    c.setFillColor(lc)
    for i, ln in enumerate(lines):
        c.drawString(bx + 4*mm, by + bh - 15*mm - i * 7.5, ln)

def draw_small_hex(hx, hy, size, fill_col, alpha=0.5):
    """Draw a small filled hexagon."""
    pts = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        pts.append((hx + size * math.cos(angle), hy + size * math.sin(angle)))
    p = c.beginPath()
    p.moveTo(*pts[0])
    for pt in pts[1:]:
        p.lineTo(*pt)
    p.close()
    c.setFillColor(fill_col)
    c.setFillAlpha(alpha)
    c.drawPath(p, fill=1, stroke=0)
    c.setFillAlpha(1.0)

def draw_hex_outline(hx, hy, size, stroke_col, alpha=0.4, lw=0.6):
    pts = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        pts.append((hx + size * math.cos(angle), hy + size * math.sin(angle)))
    p = c.beginPath()
    p.moveTo(*pts[0])
    for pt in pts[1:]:
        p.lineTo(*pt)
    p.close()
    c.setStrokeColor(stroke_col)
    c.setStrokeAlpha(alpha)
    c.setLineWidth(lw)
    c.drawPath(p, fill=0, stroke=1)
    c.setStrokeAlpha(1.0)

# ─── TITLE BLOCK (top) ────────────────────────────────────────────────────────
# Thin rule at top
c.setStrokeColor(TEAL)
c.setStrokeAlpha(0.7)
c.setLineWidth(0.5)
c.line(18*mm, H - 14*mm, W - 18*mm, H - 14*mm)
c.setStrokeAlpha(1.0)

c.setFillColor(TEAL_LT)
c.setFont('Poppins-Light', 7)
c.drawString(18*mm, H - 12*mm, 'FRAMEWORK FOR MULTI-SCALE ECOSYSTEM RESILIENCE ASSESSMENT')

# Main title
c.setFillColor(CREAM)
c.setFont('Lora', 17)
c.drawString(18*mm, H - 25*mm, 'Imaging Technologies & Ecological Monitoring')

c.setFillColor(WHITE_DIM)
c.setFont('Poppins-Light', 8)
c.drawString(18*mm, H - 34*mm, 'Marine Protected Areas · Tuscan Archipelago · Essential Ocean Variables')

# Right side: lab/affiliation stub
c.setFillColor(TEAL)
c.setFont('Poppins-Light', 6)
tx = W - 18*mm
c.drawRightString(tx, H - 12*mm, 'PhD RESEARCH FRAMEWORK')
c.setFillColor(WHITE_DIM)
c.drawRightString(tx, H - 19*mm, 'Remote Sensing  ·  Computer Vision  ·  3D Photogrammetry')

# Thin rule below title
c.setStrokeColor(TEAL)
c.setStrokeAlpha(0.4)
c.setLineWidth(0.4)
c.line(18*mm, H - 40*mm, W - 18*mm, H - 40*mm)
c.setStrokeAlpha(1.0)

# ─── SCALE AXIS (left vertical bar) ──────────────────────────────────────────
axis_x = 24*mm
axis_top = H - 50*mm
axis_bot = 22*mm
c.setStrokeColor(TEAL_LT)
c.setStrokeAlpha(0.35)
c.setLineWidth(0.8)
c.line(axis_x, axis_bot, axis_x, axis_top)
c.setStrokeAlpha(1.0)

scales = [
    ('MACRO', 0.88, 'Archipelago / Seascape'),
    ('MESO',  0.55, 'Habitat / Reef Unit'),
    ('MICRO', 0.22, 'Organism / Colony'),
]
for label, frac, sublabel in scales:
    sy = axis_bot + frac * (axis_top - axis_bot)
    c.setStrokeColor(TEAL_LT)
    c.setStrokeAlpha(0.5)
    c.setLineWidth(0.5)
    c.line(axis_x - 3*mm, sy, axis_x + 3*mm, sy)
    c.setFillColor(TEAL_LT)
    c.setFont('Poppins-Bold', 6)
    c.drawRightString(axis_x - 4*mm, sy - 2, label)
    c.setFillColor(WHITE_DIM)
    c.setFont('Poppins-Light', 5)
    c.drawRightString(axis_x - 4*mm, sy - 8, sublabel)

c.setFillColor(TEAL_LT)
c.setFont('Poppins-Light', 6)
c.saveState()
c.translate(axis_x - 10*mm, (axis_bot + axis_top) / 2)
c.rotate(90)
c.drawCentredString(0, 0, 'SPATIAL SCALE')
c.restoreState()

# ─── THREE IMAGING TECHNOLOGY NODES ──────────────────────────────────────────
# Positions: arranged in a triangle around center
tech_r_outer = 28*mm
tech_r_inner = 20*mm

tech_nodes = [
    {
        'id': 'CV',
        'label': 'COMPUTER VISION',
        'sub': 'Deep Learning · Image Segmentation',
        'cx': W * 0.28,
        'cy': H * 0.70,
        'fill': TEAL,
        'stroke': TEAL_LT,
        'dot': TEAL_LT,
    },
    {
        'id': 'RS',
        'label': 'REMOTE SENSING',
        'sub': 'Satellite · UAV · Multispectral',
        'cx': W * 0.72,
        'cy': H * 0.70,
        'fill': CERULEAN,
        'stroke': CERULEAN_LT,
        'dot': CERULEAN_LT,
    },
    {
        'id': '3D',
        'label': '3D PHOTOGRAMMETRY',
        'sub': 'SfM · Point Cloud · Structural Complexity',
        'cx': W * 0.50,
        'cy': H * 0.40,
        'fill': HexColor('#5C4A2A'),
        'stroke': SAND_LT,
        'dot': SAND,
    },
]

# Draw connecting lines between tech nodes first (behind nodes)
pairs = [(0, 1), (1, 2), (0, 2)]
for i, j in pairs:
    n1 = tech_nodes[i]
    n2 = tech_nodes[j]
    draw_dashed_line(n1['cx'], n1['cy'], n2['cx'], n2['cy'],
                     TEAL_LT, alpha=0.25, dash=(3, 6), lw=0.6)

# Draw nodes
for tn in tech_nodes:
    # Outer glow ring
    c.setStrokeColor(tn['stroke'])
    c.setStrokeAlpha(0.15)
    c.setLineWidth(8)
    c.circle(tn['cx'], tn['cy'], tech_r_outer + 4*mm, fill=0, stroke=1)
    c.setStrokeAlpha(1.0)
    c.setLineWidth(1.0)
    # Outer ring
    c.setFillColor(BG)
    c.setStrokeColor(tn['stroke'])
    c.setStrokeAlpha(0.85)
    c.setLineWidth(0.8)
    c.circle(tn['cx'], tn['cy'], tech_r_outer, fill=1, stroke=1)
    # Inner fill
    c.setFillColor(tn['fill'])
    c.setFillAlpha(0.6)
    c.circle(tn['cx'], tn['cy'], tech_r_inner, fill=1, stroke=0)
    c.setFillAlpha(1.0)
    # Second ring
    c.setStrokeColor(tn['stroke'])
    c.setStrokeAlpha(0.4)
    c.setLineWidth(0.4)
    c.circle(tn['cx'], tn['cy'], tech_r_inner + 3*mm, fill=0, stroke=1)
    c.setStrokeAlpha(1.0)
    # Label
    c.setFillColor(CREAM)
    c.setFont('Poppins-Bold', 7.5)
    tw = c.stringWidth(tn['label'], 'Poppins-Bold', 7.5)
    c.drawString(tn['cx'] - tw/2, tn['cy'] + 2, tn['label'])
    c.setFillColor(WHITE_DIM)
    c.setFont('Poppins-Light', 5.5)
    tw2 = c.stringWidth(tn['sub'], 'Poppins-Light', 5.5)
    c.drawString(tn['cx'] - tw2/2, tn['cy'] - 7, tn['sub'])
    # Small dot cluster inside
    for angle_deg in range(0, 360, 45):
        angle_r = math.radians(angle_deg)
        dx = 10*mm * math.cos(angle_r)
        dy = 10*mm * math.sin(angle_r)
        c.setFillColor(tn['dot'])
        c.setFillAlpha(0.4)
        c.circle(tn['cx'] + dx, tn['cy'] + dy, 1.2*mm, fill=1, stroke=0)
    c.setFillAlpha(1.0)

# ─── CENTER NEXUS (Ecosystem Resilience) ──────────────────────────────────────
nex_x = W * 0.50
nex_y = H * 0.57
nex_r = 18*mm

# Glow
c.setStrokeColor(GOLD)
c.setStrokeAlpha(0.10)
c.setLineWidth(16)
c.circle(nex_x, nex_y, nex_r + 8*mm, fill=0, stroke=1)
c.setStrokeAlpha(1.0)

c.setFillColor(HexColor('#1A1400'))
c.setStrokeColor(GOLD)
c.setStrokeAlpha(0.9)
c.setLineWidth(1.2)
c.circle(nex_x, nex_y, nex_r, fill=1, stroke=1)
c.setStrokeAlpha(1.0)

c.setFillColor(HexColor('#3D2E00'))
c.setFillAlpha(0.7)
c.circle(nex_x, nex_y, nex_r - 5*mm, fill=1, stroke=0)
c.setFillAlpha(1.0)

c.setStrokeColor(GOLD)
c.setStrokeAlpha(0.3)
c.setLineWidth(0.4)
c.circle(nex_x, nex_y, nex_r - 2*mm, fill=0, stroke=1)
c.setStrokeAlpha(1.0)

c.setFillColor(GOLD)
c.setFont('Poppins-Bold', 8)
tw = c.stringWidth('ECOSYSTEM', 'Poppins-Bold', 8)
c.drawString(nex_x - tw/2, nex_y + 4, 'ECOSYSTEM')
tw2 = c.stringWidth('RESILIENCE', 'Poppins-Bold', 8)
c.drawString(nex_x - tw2/2, nex_y - 5, 'RESILIENCE')
c.setFillColor(WHITE_DIM)
c.setFont('Poppins-Light', 5)
tw3 = c.stringWidth('Temporal · Spatial · Functional', 'Poppins-Light', 5)
c.drawString(nex_x - tw3/2, nex_y - 13, 'Temporal · Spatial · Functional')

# Lines from tech nodes to nexus
for tn in tech_nodes:
    dx = nex_x - tn['cx']
    dy = nex_y - tn['cy']
    dist = math.sqrt(dx**2 + dy**2)
    # start at edge of tech node, end at edge of nexus
    sx = tn['cx'] + dx / dist * (tech_r_outer + 1*mm)
    sy = tn['cy'] + dy / dist * (tech_r_outer + 1*mm)
    ex = nex_x - dx / dist * (nex_r + 1*mm)
    ey = nex_y - dy / dist * (nex_r + 1*mm)
    draw_arrow_line(sx, sy, ex, ey, GOLD, alpha=0.55, lw=0.9)

# ─── MPA NODE (bottom center) ─────────────────────────────────────────────────
mpa_x = W * 0.50
mpa_y = H * 0.13
mpa_r = 14*mm

c.setFillColor(HexColor('#001428'))
c.setStrokeColor(CERULEAN_LT)
c.setStrokeAlpha(0.8)
c.setLineWidth(1.0)
c.circle(mpa_x, mpa_y, mpa_r, fill=1, stroke=1)
c.setStrokeAlpha(1.0)
c.setFillColor(CERULEAN)
c.setFillAlpha(0.5)
c.circle(mpa_x, mpa_y, mpa_r - 4*mm, fill=1, stroke=0)
c.setFillAlpha(1.0)
c.setFillColor(CREAM)
c.setFont('Poppins-Bold', 7)
tw = c.stringWidth('TUSCAN MPA', 'Poppins-Bold', 7)
c.drawString(mpa_x - tw/2, mpa_y + 2, 'TUSCAN MPA')
c.setFillColor(WHITE_DIM)
c.setFont('Poppins-Light', 5)
tw2 = c.stringWidth('Archipelago · Seascape', 'Poppins-Light', 5)
c.drawString(mpa_x - tw2/2, mpa_y - 6, 'Archipelago · Seascape')

# Connect MPA to resilience nexus
dx = nex_x - mpa_x
dy = nex_y - mpa_y
dist = math.sqrt(dx**2 + dy**2)
sx = mpa_x + dx/dist * (mpa_r + 1*mm)
sy = mpa_y + dy/dist * (mpa_r + 1*mm)
ex = nex_x - dx/dist * (nex_r + 1*mm)
ey = nex_y - dy/dist * (nex_r + 1*mm)
draw_arrow_line(sx, sy, ex, ey, CERULEAN_LT, alpha=0.55, lw=0.8)

# ─── EOV NODE (right) ─────────────────────────────────────────────────────────
eov_x = W * 0.87
eov_y = H * 0.50
eov_r = 14*mm

c.setFillColor(HexColor('#1A0A00'))
c.setStrokeColor(AMBER)
c.setStrokeAlpha(0.8)
c.setLineWidth(1.0)
c.circle(eov_x, eov_y, eov_r, fill=1, stroke=1)
c.setStrokeAlpha(1.0)
c.setFillColor(HexColor('#7A3A00'))
c.setFillAlpha(0.5)
c.circle(eov_x, eov_y, eov_r - 4*mm, fill=1, stroke=0)
c.setFillAlpha(1.0)
c.setFillColor(AMBER)
c.setFont('Poppins-Bold', 7)
tw = c.stringWidth('EOVs', 'Poppins-Bold', 7)
c.drawString(eov_x - tw/2, eov_y + 3, 'EOVs')
c.setFillColor(WHITE_DIM)
c.setFont('Poppins-Light', 5)
tw2 = c.stringWidth('Essential Ocean', 'Poppins-Light', 5)
c.drawString(eov_x - tw2/2, eov_y - 4, 'Essential Ocean')
tw3 = c.stringWidth('Variables', 'Poppins-Light', 5)
c.drawString(eov_x - tw3/2, eov_y - 10, 'Variables')

# Connect EOV to nexus
dx = nex_x - eov_x
dy = nex_y - eov_y
dist = math.sqrt(dx**2 + dy**2)
sx = eov_x + dx/dist * (eov_r + 1*mm)
sy = eov_y + dy/dist * (eov_r + 1*mm)
ex = nex_x - dx/dist * (nex_r + 1*mm)
ey = nex_y - dy/dist * (nex_r + 1*mm)
draw_dashed_line(sx, sy, ex, ey, AMBER, alpha=0.7, dash=(5, 4), lw=0.9)

# ─── METRICS BOXES (below each tech node) ─────────────────────────────────────
box_h = 46*mm
box_w = 52*mm

# CV metrics
cv = tech_nodes[0]
label_box(
    cv['cx'] - box_w/2, cv['cy'] - tech_r_outer - 2*mm - box_h,
    box_w, box_h,
    'DERIVED METRICS',
    ['Species ID & abundance', 'Coral cover %', 'Bleaching severity index',
     'Behavioural indicators', 'Benthic composition map'],
    TEAL_LT, bg_col=TEAL_DK
)

# RS metrics
rs = tech_nodes[1]
label_box(
    rs['cx'] - box_w/2, rs['cy'] - tech_r_outer - 2*mm - box_h,
    box_w, box_h,
    'DERIVED METRICS',
    ['NDVI / SAM indices', 'Seagrass extent & health',
     'SST & thermal anomaly', 'Turbidity & water clarity',
     'Change detection (multi-temporal)'],
    CERULEAN_LT, bg_col=CERULEAN
)

# 3D metrics
pg = tech_nodes[2]
label_box(
    pg['cx'] - box_w/2, pg['cy'] + tech_r_outer + 2*mm,
    box_w, box_h,
    'DERIVED METRICS',
    ['Structural complexity (VRM)', 'Rugosity index', 'Biomass proxy (volume)',
     'Colony height & growth rate', '3D habitat mapping'],
    SAND_LT, bg_col=HexColor('#3A2C10')
)

# ─── EOV DETAIL BOX ───────────────────────────────────────────────────────────
eov_box_w = 38*mm
eov_box_h = 60*mm
label_box(
    eov_x - eov_box_w/2, eov_y + eov_r + 4*mm,
    eov_box_w, eov_box_h,
    'BIOLOGICAL EOVs',
    ['Hard coral cover/comp.', 'Seagrass cover/comp.',
     'Macroalgae cover/comp.', 'Fish community comp.',
     'Microbe community comp.', 'Marine turtles abundance'],
    AMBER, bg_col=HexColor('#1A0800')
)

# ─── SCALE INTEGRATION ARROWS (linking tech nodes to scales) ──────────────────
# Small tick-marks along the scale axis for each technology
def tech_to_scale(tech_cx, tech_cy, scale_y, col, alpha=0.3):
    ex = axis_x + 3*mm
    draw_dashed_line(tech_cx, tech_cy, ex + 10*mm, scale_y, col, alpha=alpha, dash=(2, 5), lw=0.5)

# CV → micro & meso
tech_to_scale(cv['cx'], cv['cy'], axis_bot + 0.22 * (axis_top - axis_bot), TEAL, alpha=0.22)
tech_to_scale(cv['cx'], cv['cy'], axis_bot + 0.55 * (axis_top - axis_bot), TEAL, alpha=0.18)
# RS → macro & meso
tech_to_scale(rs['cx'], rs['cy'], axis_bot + 0.88 * (axis_top - axis_bot), CERULEAN_LT, alpha=0.22)
tech_to_scale(rs['cx'], rs['cy'], axis_bot + 0.55 * (axis_top - axis_bot), CERULEAN_LT, alpha=0.18)
# 3D → micro & meso
tech_to_scale(pg['cx'], pg['cy'], axis_bot + 0.22 * (axis_top - axis_bot), SAND, alpha=0.22)
tech_to_scale(pg['cx'], pg['cy'], axis_bot + 0.55 * (axis_top - axis_bot), SAND, alpha=0.18)

# ─── HEXAGONAL FIELD (bottom left – texture representing habitat cells) ────────
hex_area_cx = 72*mm
hex_area_cy = 52*mm
hex_size = 5.5*mm
cols_hex = ['TEAL', 'CERULEAN', 'SAND', 'TEAL', 'CERULEAN', 'SAND']
fill_cols = [TEAL, CERULEAN, HexColor('#4A3A1A'), TEAL_DK, CERULEAN, SAND]
alphas_base = [0.35, 0.25, 0.20, 0.15, 0.10, 0.08]
for row in range(-4, 5):
    for col in range(-4, 5):
        hx = hex_area_cx + col * hex_size * 1.732
        hy = hex_area_cy + row * hex_size * 2 + (col % 2) * hex_size
        dist_from_c = math.sqrt((hx - hex_area_cx)**2 + (hy - hex_area_cy)**2)
        fade = max(0, 1 - dist_from_c / (28*mm))
        idx = abs(row + col) % len(fill_cols)
        draw_small_hex(hx, hy, hex_size * 0.85, fill_cols[idx], alpha=alphas_base[idx] * fade)
        draw_hex_outline(hx, hy, hex_size * 0.85, TEAL_LT, alpha=0.12 * fade)

# Small label
c.setFillColor(WHITE_DIM)
c.setFont('Poppins-Light', 5.5)
c.drawCentredString(hex_area_cx, hex_area_cy - 36*mm, 'BENTHIC HABITAT MOSAIC')
c.drawCentredString(hex_area_cx, hex_area_cy - 42*mm, '(cell-level resolution · CV classification)')

# ─── SCALE INTEGRATION MATRIX (right side, below EOV) ─────────────────────────
matrix_x = eov_x - 24*mm
matrix_y = 22*mm
rows = ['MACRO', 'MESO', 'MICRO']
cols_mat = ['CV', 'RS', '3D']
cell_w = 16*mm
cell_h = 10*mm
# Presence map (1 = strong, 0.5 = partial)
presence = [
    [0.5, 1.0, 0.5],   # MACRO: CV partial, RS full, 3D partial
    [1.0, 1.0, 1.0],   # MESO: all full
    [1.0, 0.5, 1.0],   # MICRO: CV full, RS partial, 3D full
]
tech_colors = [TEAL, CERULEAN_LT, SAND]
for ri, row in enumerate(rows):
    for ci, col in enumerate(cols_mat):
        cx_cell = matrix_x + ci * cell_w + cell_w/2
        cy_cell = matrix_y + ri * cell_h + cell_h/2
        val = presence[ri][ci]
        tc = tech_colors[ci]
        c.setFillColor(tc)
        c.setFillAlpha(val * 0.55)
        c.roundRect(matrix_x + ci*cell_w + 1, matrix_y + ri*cell_h + 1,
                    cell_w - 2, cell_h - 2, 1.5*mm, fill=1, stroke=0)
        c.setFillAlpha(1.0)
        c.setFillColor(CREAM if val == 1.0 else WHITE_DIM)
        c.setFont('Poppins-Light', 4.5)
        sym = '●' if val == 1.0 else '◐'
        sw = c.stringWidth(sym, 'Poppins-Light', 4.5)
        c.drawString(cx_cell - sw/2, cy_cell - 2, sym)

# Column labels
for ci, col in enumerate(cols_mat):
    c.setFillColor(tech_colors[ci])
    c.setFont('Poppins-Bold', 5.5)
    cx2 = matrix_x + ci * cell_w + cell_w/2
    tw = c.stringWidth(col, 'Poppins-Bold', 5.5)
    c.drawString(cx2 - tw/2, matrix_y + 3*cell_h + 3, col)

# Row labels
for ri, row in enumerate(rows):
    c.setFillColor(WHITE_DIM)
    c.setFont('Poppins-Light', 5)
    c.drawRightString(matrix_x - 2, matrix_y + ri*cell_h + cell_h/2 - 2, row)

# Matrix title
c.setFillColor(WHITE_DIM)
c.setFont('Poppins-Light', 5)
c.drawCentredString(matrix_x + 1.5*cell_w, matrix_y + 3*cell_h + 9, 'SCALE × TECHNOLOGY')

# ─── INNOVATION CALLOUTS ──────────────────────────────────────────────────────
# Small annotation lines for innovations
innovations = [
    (W * 0.28, H * 0.84, 'INNOVATION: Automated species ID', 'Transfer learning · foundation models', TEAL_LT),
    (W * 0.72, H * 0.84, 'INNOVATION: Hyperspectral fusion', 'Satellite × UAV data integration', CERULEAN_LT),
    (W * 0.50, H * 0.30, 'INNOVATION: High-res 3D mapping', 'Photogrammetry at colony scale', SAND_LT),
]
for ix, iy, title, sub, col in innovations:
    c.setFillColor(col)
    c.setFont('Poppins-Bold', 5.5)
    tw = c.stringWidth(title, 'Poppins-Bold', 5.5)
    c.drawString(ix - tw/2, iy + 4, title)
    c.setFillColor(WHITE_DIM)
    c.setFont('Poppins-Light', 5)
    tw2 = c.stringWidth(sub, 'Poppins-Light', 5)
    c.drawString(ix - tw2/2, iy - 3, sub)

# ─── FLOW ARROWS: tech → resilience pathways labeled ──────────────────────────
# Mid-point labels on the tech→nexus arrows
mid_labels = [
    (tech_nodes[0], 'spectral classification\n& change detection'),
    (tech_nodes[1], 'landscape metrics\n& seascape connectivity'),
    (tech_nodes[2], 'structural complexity\n& biomass estimation'),
]
for tn, lbl in mid_labels:
    mx = (tn['cx'] + nex_x) / 2
    my = (tn['cy'] + nex_y) / 2
    lines = lbl.split('\n')
    for i, l in enumerate(lines):
        c.setFillColor(WHITE_DIM)
        c.setFont('Poppins-Light', 4.8)
        tw = c.stringWidth(l, 'Poppins-Light', 4.8)
        c.drawString(mx - tw/2 + 3*mm, my + 4 - i*7, l)

# ─── RESILIENCE DIMENSIONS (spokes from nexus) ────────────────────────────────
dim_labels = [
    (270, 'RESISTANCE'),
    (210, 'RECOVERY'),
    (330, 'REDUNDANCY'),
]
for deg, dlabel in dim_labels:
    rad = math.radians(deg)
    ex = nex_x + math.cos(rad) * (nex_r + 28*mm)
    ey = nex_y + math.sin(rad) * (nex_r + 28*mm)
    sx = nex_x + math.cos(rad) * (nex_r + 1*mm)
    sy = nex_y + math.sin(rad) * (nex_r + 1*mm)
    draw_dashed_line(sx, sy, ex, ey, GOLD, alpha=0.45, dash=(3, 4), lw=0.7)
    c.setFillColor(GOLD)
    c.setFont('Poppins-Light', 5.5)
    tw = c.stringWidth(dlabel, 'Poppins-Light', 5.5)
    # offset label to not overlap
    lx = ex + math.cos(rad) * 3*mm
    ly = ey + math.sin(rad) * 3*mm
    c.drawString(lx - tw/2, ly - 2, dlabel)

# ─── BOTTOM CAPTION / METHODOLOGY NOTE ───────────────────────────────────────
c.setStrokeColor(TEAL)
c.setStrokeAlpha(0.3)
c.setLineWidth(0.4)
c.line(18*mm, 14*mm, W - 18*mm, 14*mm)
c.setStrokeAlpha(1.0)

c.setFillColor(WHITE_DIM)
c.setFont('Poppins-Light', 5.5)
caption = ('PhD Research Framework  ·  Multi-scale integration of Computer Vision, Remote Sensing & 3D Photogrammetry  '
           '·  Tuscan Archipelago Marine Protected Area  ·  Biological Essential Ocean Variables (B-EOVs)')
tw = c.stringWidth(caption, 'Poppins-Light', 5.5)
c.drawString((W - tw)/2, 7*mm, caption)

# ─── CORNER FIDUCIALS ─────────────────────────────────────────────────────────
for fx, fy in [(18*mm, 18*mm), (W - 18*mm, 18*mm),
               (18*mm, H - 18*mm), (W - 18*mm, H - 18*mm)]:
    c.setStrokeColor(TEAL_LT)
    c.setStrokeAlpha(0.3)
    c.setLineWidth(0.5)
    c.circle(fx, fy, 2.5*mm, fill=0, stroke=1)
    c.circle(fx, fy, 1*mm, fill=0, stroke=1)
c.setStrokeAlpha(1.0)

# ─── SAVE ─────────────────────────────────────────────────────────────────────
c.save()
print(f'Saved to {OUT}')