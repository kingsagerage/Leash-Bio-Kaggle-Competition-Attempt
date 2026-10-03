from pathlib import Path
from html import escape
import math
import cairosvg

OUT = Path(__file__).resolve().parents[1] / 'assets'
OUT.mkdir(parents=True, exist_ok=True)
NAVY = '#102A43'
INK = '#193449'
MUTED = '#536B7B'
BORDER = '#CEDDE5'
GREEN = '#087F70'
MINT = '#E7F5F0'
PURPLE = '#6955B7'
LAVENDER = '#F0ECFA'
BLUE = '#2E679A'
PALEBLUE = '#EAF3FA'
AMBER = '#9B6417'
PALEAMBER = '#FFF5E3'
BG = '#F7FAFC'

class SVG:
    def __init__(self, width, height, title, desc, dark=False):
        self.w, self.h = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',
                      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#829BA9"/></marker></defs>']
        self.rect(0, 0, width, height, NAVY if dark else BG, rx=22)
    def rect(self, x, y, w, h, fill, stroke='none', rx=12, sw=1):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def text(self, x, y, txt, size=22, color=INK, weight=400, anchor='start', spacing=None):
        extra = '' if spacing is None else f' letter-spacing="{spacing}"'
        self.parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans,Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{extra}>{escape(str(txt))}</text>')
    def lines(self, x, y, rows, size=22, color=MUTED, step=31, weight=400):
        for i, row in enumerate(rows):
            self.text(x, y+i*step, row, size, color, weight)
    def line(self, x1,y1,x2,y2,color=BORDER,sw=2,arrow=False):
        self.parts.append(f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{color}" stroke-width="{sw}" fill="none"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
    def path(self, d, color=BORDER, sw=2, arrow=True):
        self.parts.append(f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="none"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
    def circle(self, x, y, r, fill, opacity=1):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" opacity="{opacity}"/>')
    def header(self, num, title, sub):
        self.text(48, 53, f'{num} / BELKA BASELINE', 16, GREEN, 700, spacing=2)
        self.text(48, 101, title, 34, NAVY, 700)
        self.text(48, 139, sub, 21, MUTED)
    def save(self, name):
        content = '\n'.join(self.parts) + '\n</svg>\n'
        path = OUT / f'{name}.svg'
        path.write_text(content, encoding='utf-8')
        cairosvg.svg2png(bytestring=content.encode(), write_to=str(OUT/f'{name}.png'))

# Header is an abstract feature-projection motif, not a chemical structure.
s = SVG(1440, 320, 'BELKA - Molecular binding prediction',
        'Leash Bio Kaggle 2024. Morgan fingerprints, protein identity, PCA, dense neural network. Abstract feature-matrix illustration, not measured data.', dark=True)
s.text(48, 54, 'LEASH BIO  /  KAGGLE 2024', 17, '#8DDFCC', 700, spacing=2)
s.text(48, 128, 'BELKA', 64, '#FFFFFF', 700)
s.text(50, 175, 'Molecular binding prediction', 32, '#E8F2F7', 500)
s.text(50, 216, 'Morgan fingerprints + protein identity + PCA + neural classifier', 20, '#BAD0DF')
for x, label, w in [(50,'RDKit',100),(166,'scikit-learn',150),(332,'TensorFlow / Keras',230),(578,'Research baseline',216)]:
    s.rect(x, 250, w, 38, '#1D3D55', '#36586E', rx=19)
    s.text(x+w/2, 275, label, 15, '#E6F5F5', 500, 'middle')
for row in range(7):
    for col in range(8):
        s.rect(965+col*23, 48+row*27, 13, 17, '#69CAB5' if (row*5+col*3)%7 < 3 else '#335268', rx=3)
s.path('M 1160 138 L 1225 138', '#6E91A6', 3)
for row in range(7):
    for col in range(3):
        s.rect(1244+col*25, 48+row*27, 14, 17, '#AD9DE1' if (row+col)%3 else '#7360AA', rx=3)
s.text(962, 281, '1,027 features', 16, '#BAD0DF')
s.text(1239, 281, '400 PCs', 16, '#D9CEFF')
s.save('hero')

# Feature extraction, including protein identity inside PCA.
s = SVG(1440, 630, 'Feature engineering pipeline',
        'Molecule SMILES becomes a 1024-bit radius-3 Morgan fingerprint. Protein name becomes three one-hot indicators. Combined 1027 inputs pass together through PCA with 400 components, then a dense classifier yields one binding score.')
s.header('01', 'From molecular structure to a binding score',
         'The documented model conditions one binary prediction on both molecule and protein.')
for x,y,w,h,fill,title,rows in [
    (48,188,320,152,MINT,'Molecular structure',['molecule_smiles','RDKit molecule object']),
    (410,188,340,152,MINT,'Morgan fingerprint',['Radius 3  |  1,024 bits','ECFP-style binary features']),
    (48,377,320,152,PALEBLUE,'Protein identity',['BRD4  /  HSA  /  sEH','protein_name']),
    (410,377,340,152,PALEBLUE,'One-hot encoding',['3 indicator columns','pandas.get_dummies']),
    (817,273,230,173,LAVENDER,'Joint features',['1,027 columns','PCA: 400 PCs']),
    (1112,273,280,173,'#FFFFFF','Dense classifier',['6 hidden layers','1 sigmoid binding score'])]:
    s.rect(x,y,w,h,fill,BORDER)
    s.text(x+22,y+40,title,23,NAVY,700)
    s.lines(x+22,y+80,rows,21)
s.line(368,264,406,264,arrow=True)
s.line(368,453,406,453,arrow=True)
s.path('M 750 264 L 783 264 L 783 319 L 813 319')
s.path('M 750 453 L 783 453 L 783 401 L 813 401')
s.line(1047,359,1108,359,arrow=True)
s.text(48,577,'id and binds are excluded from X. Building-block SMILES are not independently featurized.',19,MUTED)
s.text(48,606,'PCA includes the protein indicators; they are not appended after dimensionality reduction.',19,MUTED)
s.save('feature_pipeline')

# Neural architecture, precise layer count.
s = SVG(1440, 563, 'Dense model architecture',
        'The declared architecture is 400 input components, hidden layers of 128, 256, 256, 256, 256, 128 ReLU units and one sigmoid output. Parameter total including biases is 314753, derived from code. Saved 200-column output conflicts with the 400-dimensional declaration.')
s.header('02','A compact, target-conditioned neural classifier',
         'Declared architecture: 400 > 128 > 256 > 256 > 256 > 256 > 128 > 1')
widths = [400,128,256,256,256,256,128,1]
for i, units in enumerate(widths):
    x = 48+i*171
    fill = LAVENDER if i==0 else MINT if i==7 else '#FFFFFF'
    s.rect(x,194,147,205,fill,BORDER)
    s.text(x+73.5,228,'PCA INPUT' if i==0 else 'OUTPUT' if i==7 else f'DENSE {i}',15,GREEN if i==7 else MUTED,700,'middle')
    s.text(x+73.5,288,f'{units:,}',43,NAVY,700,'middle')
    s.text(x+73.5,320,'components' if i==0 else 'sigmoid' if i==7 else 'ReLU',18,MUTED,400,'middle')
    for j in range(5 if i not in (0,7) else 3 if i==0 else 1):
        n=5 if i not in (0,7) else 3 if i==0 else 1
        s.circle(x+73.5+(j-(n-1)/2)*16,360,4.5,PURPLE if i==0 else GREEN if i==7 else BLUE,0.85)
    if i<7:
        s.line(x+149,292,x+167,292,arrow=True)
s.text(48,447,'314,753 trainable parameters',25,NAVY,700)
s.text(535,447,'Derived from layer widths and biases; not a saved-model measurement.',19,MUTED)
s.rect(48,473,1344,56,PALEAMBER,rx=10)
s.text(68,508,'Source consistency note: PCA and Dense declare 400 inputs; a saved X_test printout shows 200.',20,AMBER)
s.save('model_architecture')

# Faithful, not silently corrected, training order.
s = SVG(1440, 880, 'Training and evaluation flow as supplied',
        'One million CSV rows are encoded. PCA is fitted to all rows before an 80/20 split. The 800000-row training partition is split again by Keras into 640000 optimization rows and 160000 validation rows. 200000 rows are used for model.evaluate. The PCA-before-split order is flagged as a validation caveat.')
s.header('03','Training flow in the supplied experiment',
         'Counts are derived for 1,000,000 retained rows; no rows removed during preprocessing.')
for x,title,rows,fill in [
    (48,'Load CSV prefix',['First 1,000,000 rows','Not a random subsample'], '#FFFFFF'),
    (503,'Build features',['1,024 bits + 3 indicators','RDKit + pandas'],MINT),
    (958,'Fit PCA on all rows',['400 components','Before train/test split'],PALEAMBER)]:
    s.rect(x,182,434,134,fill,BORDER)
    s.text(x+22,220,title,23,NAVY,700)
    s.lines(x+22,257,rows,20,step=29)
s.line(482,249,498,249,arrow=True)
s.line(937,249,953,249,arrow=True)
s.path('M 1175 316 L 1175 345 L 720 345 L 720 373')
s.rect(490,377,460,102,LAVENDER,BORDER)
s.text(720,417,'80 / 20 row-wise split',26,NAVY,700,'middle')
s.text(720,451,'test_size=0.2  |  random_state=42',20,MUTED,400,'middle')
s.path('M 605 479 L 605 515 L 417 515 L 417 545')
s.path('M 835 479 L 835 515 L 1175 515 L 1175 545')
s.rect(48,549,738,156,'#FFFFFF',BORDER)
s.text(70,587,'800,000 rows passed to model.fit',24,NAVY,700)
s.rect(70,610,359,72,MINT,rx=10)
s.text(249.5,638,'640,000 optimization rows',19,GREEN,700,'middle')
s.text(249.5,665,'64% of loaded rows',17,MUTED,400,'middle')
s.rect(444,610,318,72,PALEBLUE,rx=10)
s.text(603,638,'160,000 validation rows',19,BLUE,700,'middle')
s.text(603,665,'validation_split=0.2',17,MUTED,400,'middle')
s.rect(958,549,434,156,'#FFFFFF',BORDER)
s.text(980,587,'200,000 test rows',24,NAVY,700)
s.lines(980,626,['model.evaluate(X_test, y_test)','Loss, accuracy, precision'],20)
s.rect(48,731,1344,57,NAVY)
s.text(70,768,'30 epochs  |  batch size 32  |  Adam 0.001  |  binary cross-entropy',22,'#FFFFFF',500)
s.text(48,823,'Validation caveat: fitted PCA has already seen validation and test features. Split before fitting',20,AMBER)
s.text(48,852,'PCA for a new unbiased evaluation. This diagram preserves the original execution order.',20,AMBER)
s.save('training_workflow')

# Source-derived summary, no invented performance charts.
s = SVG(1440, 494, 'Documented data snapshot',
        'The supplied output reports 997424 negative and 2576 positive rows out of one million. The positive rate is 0.2576 percent. The PCA output reports 73.56 percent explained variance, but its component count cannot be reconciled with the mixed output. No predictive evaluation values are present.')
s.header('04','What the saved output actually reports',
         'Data and preprocessing evidence only - not model accuracy or a leaderboard result.')
cards=[(48,423,'1,000,000','molecule-protein rows',['Not a count of unique molecules.'],NAVY,'#FFFFFF'),
       (508,424,'2,576','positive binding labels',['997,424 labels are negative.'],GREEN,MINT),
       (969,423,'0.2576%','positive-class share',['Calculated from the label counts.'],PURPLE,LAVENDER)]
for x,w,big,sub,rows,color,fill in cards:
    s.rect(x,183,w,154,fill,BORDER)
    s.text(x+22,237,big,39,color,700)
    s.text(x+22,274,sub,22,NAVY,600)
    s.lines(x+22,310,rows,18,MUTED)
s.rect(48,365,1344,93,PALEAMBER,rx=12)
s.text(70,401,'PCA printout: 73.56% explained variance',23,AMBER,700)
s.text(70,435,'Saved-output value only; the 200-vs-400 component discrepancy prevents clean run attribution.',19,AMBER)
s.save('data_snapshot')

print('Created', len(list(OUT.glob('*.svg'))), 'SVGs and', len(list(OUT.glob('*.png'))), 'PNGs')
