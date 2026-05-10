
# ── Imports ───────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

from scipy.ndimage import gaussian_filter, rotate

from sklearn.neural_network    import MLPClassifier
from sklearn.svm               import SVC
from sklearn.ensemble          import RandomForestClassifier
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.decomposition     import PCA
from sklearn.metrics           import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.pipeline          import Pipeline

np.random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 65)
print("   HANDWRITTEN CHARACTER RECOGNITION — FULL ML PIPELINE")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
IMG_SIZE      = 28          # 28×28 pixels (MNIST standard)
N_PER_CLASS   = 120         # samples per character class
DIGITS        = list('0123456789')
LETTERS       = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
ALL_CLASSES   = DIGITS + LETTERS   # 36 classes total

print(f"\n   Image size   : {IMG_SIZE}×{IMG_SIZE} pixels")
print(f"   Classes      : {len(DIGITS)} digits + {len(LETTERS)} letters = {len(ALL_CLASSES)} total")
print(f"   Samples/class: {N_PER_CLASS}")
print(f"   Total samples: {len(ALL_CLASSES) * N_PER_CLASS}")

# ══════════════════════════════════════════════════════════════
# STEP 1 ── SYNTHETIC CHARACTER IMAGE GENERATION
#           Mimics MNIST/EMNIST image statistics
# ══════════════════════════════════════════════════════════════
def make_stroke(canvas, x0, y0, x1, y1, thickness=2):
    """Draw a line stroke on canvas using Bresenham-style interpolation."""
    steps = max(abs(x1-x0), abs(y1-y0), 1) * 3
    for t in np.linspace(0, 1, steps):
        x = int(x0 + t*(x1-x0))
        y = int(y0 + t*(y1-y0))
        for dx in range(-thickness, thickness+1):
            for dy in range(-thickness, thickness+1):
                nx, ny = x+dx, y+dy
                if 0 <= nx < IMG_SIZE and 0 <= ny < IMG_SIZE:
                    canvas[ny, nx] = min(1.0, canvas[ny, nx] + 0.6)

def generate_digit(char):
    """Generate a synthetic handwritten digit image."""
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    c = IMG_SIZE // 2
    r = IMG_SIZE // 2 - 4
    t = 2

    strokes = {
        '0': [[(c-6,c-10),(c+6,c-10)],[(c+6,c-10),(c+8,c)],
              [(c+8,c),(c+6,c+10)],[(c+6,c+10),(c-6,c+10)],
              [(c-6,c+10),(c-8,c)],[(c-8,c),(c-6,c-10)]],
        '1': [[(c,c-10),(c,c+10)],[(c-3,c-7),(c,c-10)]],
        '2': [[(c-6,c-10),(c+6,c-10)],[(c+6,c-10),(c+6,c)],
              [(c+6,c),(c-6,c)],[(c-6,c),(c-6,c+10)],
              [(c-6,c+10),(c+6,c+10)]],
        '3': [[(c-6,c-10),(c+6,c-10)],[(c+6,c-10),(c+6,c)],
              [(c-4,c),(c+6,c)],[(c+6,c),(c+6,c+10)],
              [(c-6,c+10),(c+6,c+10)]],
        '4': [[(c-6,c-10),(c-6,c)],[(c-6,c),(c+6,c)],
              [(c+6,c-10),(c+6,c+10)]],
        '5': [[(c+6,c-10),(c-6,c-10)],[(c-6,c-10),(c-6,c)],
              [(c-6,c),(c+6,c)],[(c+6,c),(c+6,c+10)],
              [(c-6,c+10),(c+6,c+10)]],
        '6': [[(c+6,c-10),(c-6,c-10)],[(c-6,c-10),(c-6,c+10)],
              [(c-6,c+10),(c+6,c+10)],[(c+6,c+10),(c+6,c)],
              [(c+6,c),(c-6,c)]],
        '7': [[(c-6,c-10),(c+6,c-10)],[(c+6,c-10),(c-2,c+10)]],
        '8': [[(c-6,c-10),(c+6,c-10)],[(c+6,c-10),(c+6,c)],
              [(c+6,c),(c-6,c)],[(c-6,c),(c-6,c-10)],
              [(c-6,c),(c-6,c+10)],[(c-6,c+10),(c+6,c+10)],
              [(c+6,c+10),(c+6,c)]],
        '9': [[(c-6,c-10),(c+6,c-10)],[(c+6,c-10),(c+6,c+10)],
              [(c-6,c-10),(c-6,c)],[(c-6,c),(c+6,c)]],
    }
    for seg in strokes.get(char, [[(c-5,c-8),(c+5,c+8)]]):
        make_stroke(canvas, seg[0][0], seg[0][1], seg[1][0], seg[1][1], t)
    return canvas

def generate_letter(char):
    """Generate a synthetic handwritten letter image."""
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    c = IMG_SIZE // 2
    t = 2
    idx = ord(char) - ord('A')

    # Each letter gets unique strokes based on its shape
    letter_strokes = {
        'A': [[(c,c-11),(c-7,c+10)],[(c,c-11),(c+7,c+10)],[(c-4,c+1),(c+4,c+1)]],
        'B': [[(c-6,c-10),(c-6,c+10)],[(c-6,c-10),(c+5,c-10)],[(c+5,c-10),(c+7,c-5)],
              [(c+7,c-5),(c+5,c)],[(c-6,c),(c+5,c)],[(c+5,c),(c+7,c+5)],
              [(c+7,c+5),(c+5,c+10)],[(c+5,c+10),(c-6,c+10)]],
        'C': [[(c+6,c-8),(c,c-11)],[(c,c-11),(c-7,c)],[(c-7,c),(c,c+11)],
              [(c,c+11),(c+6,c+8)]],
        'D': [[(c-6,c-10),(c-6,c+10)],[(c-6,c-10),(c+4,c-10)],
              [(c+4,c-10),(c+8,c)],[(c+8,c),(c+4,c+10)],[(c+4,c+10),(c-6,c+10)]],
        'E': [[(c-6,c-10),(c-6,c+10)],[(c-6,c-10),(c+7,c-10)],
              [(c-6,c),(c+5,c)],[(c-6,c+10),(c+7,c+10)]],
        'F': [[(c-6,c-10),(c-6,c+10)],[(c-6,c-10),(c+7,c-10)],[(c-6,c),(c+5,c)]],
        'G': [[(c+6,c-8),(c,c-11)],[(c,c-11),(c-7,c)],[(c-7,c),(c,c+11)],
              [(c,c+11),(c+7,c+5)],[(c+7,c+5),(c+7,c)],[(c+7,c),(c+2,c)]],
        'H': [[(c-6,c-10),(c-6,c+10)],[(c+6,c-10),(c+6,c+10)],[(c-6,c),(c+6,c)]],
        'I': [[(c,c-10),(c,c+10)],[(c-4,c-10),(c+4,c-10)],[(c-4,c+10),(c+4,c+10)]],
        'J': [[(c+4,c-10),(c+4,c+6)],[(c+4,c+6),(c,c+11)],[(c,c+11),(c-4,c+8)]],
        'K': [[(c-6,c-10),(c-6,c+10)],[(c-6,c),(c+7,c-10)],[(c-6,c),(c+7,c+10)]],
        'L': [[(c-6,c-10),(c-6,c+10)],[(c-6,c+10),(c+7,c+10)]],
        'M': [[(c-7,c+10),(c-7,c-10)],[(c-7,c-10),(c,c+2)],
              [(c,c+2),(c+7,c-10)],[(c+7,c-10),(c+7,c+10)]],
        'N': [[(c-6,c+10),(c-6,c-10)],[(c-6,c-10),(c+6,c+10)],[(c+6,c+10),(c+6,c-10)]],
        'O': [[(c,c-11),(c+7,c)],[(c+7,c),(c,c+11)],[(c,c+11),(c-7,c)],[(c-7,c),(c,c-11)]],
        'P': [[(c-6,c-10),(c-6,c+10)],[(c-6,c-10),(c+5,c-10)],
              [(c+5,c-10),(c+7,c-5)],[(c+7,c-5),(c+5,c)],[(c+5,c),(c-6,c)]],
        'Q': [[(c,c-11),(c+7,c)],[(c+7,c),(c,c+11)],[(c,c+11),(c-7,c)],
              [(c-7,c),(c,c-11)],[(c+2,c+6),(c+8,c+12)]],
        'R': [[(c-6,c-10),(c-6,c+10)],[(c-6,c-10),(c+5,c-10)],
              [(c+5,c-10),(c+7,c-5)],[(c+7,c-5),(c+5,c)],
              [(c+5,c),(c-6,c)],[(c-2,c),(c+7,c+10)]],
        'S': [[(c+6,c-8),(c,c-11)],[(c,c-11),(c-6,c-5)],[(c-6,c-5),(c+6,c+5)],
              [(c+6,c+5),(c,c+11)],[(c,c+11),(c-6,c+8)]],
        'T': [[(c-7,c-10),(c+7,c-10)],[(c,c-10),(c,c+10)]],
        'U': [[(c-6,c-10),(c-6,c+6)],[(c-6,c+6),(c,c+11)],
              [(c,c+11),(c+6,c+6)],[(c+6,c+6),(c+6,c-10)]],
        'V': [[(c-7,c-10),(c,c+11)],[(c,c+11),(c+7,c-10)]],
        'W': [[(c-8,c-10),(c-4,c+10)],[(c-4,c+10),(c,c+2)],
              [(c,c+2),(c+4,c+10)],[(c+4,c+10),(c+8,c-10)]],
        'X': [[(c-7,c-10),(c+7,c+10)],[(c+7,c-10),(c-7,c+10)]],
        'Y': [[(c-7,c-10),(c,c)],[(c+7,c-10),(c,c)],[(c,c),(c,c+10)]],
        'Z': [[(c-7,c-10),(c+7,c-10)],[(c+7,c-10),(c-7,c+10)],[(c-7,c+10),(c+7,c+10)]],
    }
    for seg in letter_strokes.get(char, [[(c-5,c-8),(c+5,c+8)]]):
        make_stroke(canvas, seg[0][0], seg[0][1], seg[1][0], seg[1][1], t)
    return canvas

def augment_image(img):
    """
    Apply random augmentations to simulate real handwriting variation:
      - Random rotation  (±15°)
      - Random translation (±2px)
      - Gaussian blur (simulates pen/pencil spread)
      - Pixel noise
    """
    # Rotation
    angle = np.random.uniform(-15, 15)
    img = rotate(img, angle, reshape=False, mode='constant', cval=0)
    # Translation
    tx, ty = np.random.randint(-2, 3, 2)
    from scipy.ndimage import shift
    img = shift(img, [ty, tx], mode='constant', cval=0)
    # Blur
    sigma = np.random.uniform(0.3, 0.9)
    img = gaussian_filter(img, sigma=sigma)
    # Noise
    img += np.random.normal(0, 0.04, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)

print("\n[1/6] Generating Synthetic Character Images …")
X_imgs, y_labels = [], []

for char in ALL_CLASSES:
    for _ in range(N_PER_CLASS):
        if char.isdigit():
            base = generate_digit(char)
        else:
            base = generate_letter(char)
        img = augment_image(base)
        X_imgs.append(img)
        y_labels.append(char)

X_imgs   = np.array(X_imgs)     # (N, 28, 28)
y_labels = np.array(y_labels)

print(f"   ✔ {X_imgs.shape[0]} images generated  |  shape: {X_imgs.shape[1:]}")
print(f"   ✔ Classes: {len(ALL_CLASSES)}  ({len(DIGITS)} digits + {len(LETTERS)} letters)")

# ══════════════════════════════════════════════════════════════
# STEP 2 ── FEATURE ENGINEERING
#   A) Raw pixel flattening      (784 features)
#   B) HOG-style gradient features (edge detection)
#   C) PCA dimensionality reduction
# ══════════════════════════════════════════════════════════════
print("\n[2/6] Feature Engineering …")

def compute_gradient_features(imgs):
    """
    HOG-inspired gradient features:
    Compute horizontal & vertical Sobel-like gradients,
    then aggregate in a grid of cells.
    """
    feats = []
    cell  = 7    # 4×4 grid of 7×7 cells
    for img in imgs:
        gx = np.gradient(img, axis=1)
        gy = np.gradient(img, axis=0)
        mag = np.sqrt(gx**2 + gy**2)
        feat = []
        for row in range(0, IMG_SIZE, cell):
            for col in range(0, IMG_SIZE, cell):
                patch = mag[row:row+cell, col:col+cell]
                feat.extend([patch.mean(), patch.std(), patch.max()])
        feats.append(feat)
    return np.array(feats, dtype=np.float32)

# Flatten pixels
X_pixels = X_imgs.reshape(len(X_imgs), -1)          # (N, 784)

# Gradient features
X_grad   = compute_gradient_features(X_imgs)         # (N, 48)

# Combine
X_combined = np.hstack([X_pixels, X_grad])           # (N, 832)
print(f"   ✔ Pixel features   : {X_pixels.shape[1]}")
print(f"   ✔ Gradient features: {X_grad.shape[1]}")
print(f"   ✔ Combined vector  : {X_combined.shape[1]}")

# PCA reduction (retain 95% variance)
pca      = PCA(n_components=0.95, random_state=42)
scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(X_combined)
X_pca    = pca.fit_transform(X_scaled)
print(f"   ✔ PCA components   : {X_pca.shape[1]} (95% variance retained)")

# Encode labels
le = LabelEncoder()
y  = le.fit_transform(y_labels)

# Train / Test split (stratified)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_pca, y, test_size=0.20, random_state=42, stratify=y
)
print(f"   Train: {X_tr.shape[0]} | Test: {X_te.shape[0]}")

# ══════════════════════════════════════════════════════════════
# STEP 3 ── MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════
print("\n[3/6] Building Models …")

MODELS = {
    # CNN-style: deep MLP with large hidden layers + ReLU
    'CNN (Deep MLP)': MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu', solver='adam',
        max_iter=300, random_state=42,
        early_stopping=True, validation_fraction=0.12,
        learning_rate_init=0.001, batch_size=64, alpha=1e-4
    ),
    # SVM with RBF kernel (excellent for image classification)
    'SVM (RBF)': SVC(
        kernel='rbf', C=10, gamma='scale',
        probability=True, random_state=42
    ),
    # Random Forest ensemble
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=20,
        min_samples_split=5, random_state=42, n_jobs=-1
    ),
}

# ══════════════════════════════════════════════════════════════
# STEP 4 ── TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════
print("\n[4/6] Training Models …\n")
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results     = {}

for name, model in MODELS.items():
    model.fit(X_tr, y_tr)
    y_pred   = model.predict(X_te)
    cv_score = cross_val_score(model, X_tr, y_tr, cv=cv_splitter,
                               scoring='accuracy', n_jobs=-1)
    results[name] = {
        'model':       model,
        'y_pred':      y_pred,
        'accuracy':    accuracy_score(y_te, y_pred),
        'f1_macro':    f1_score(y_te, y_pred, average='macro'),
        'f1_weighted': f1_score(y_te, y_pred, average='weighted'),
        'cv_mean':     cv_score.mean(),
        'cv_std':      cv_score.std(),
    }
    print(f"   ✔ {name:<22}  Acc={results[name]['accuracy']:.4f}  "
          f"F1={results[name]['f1_macro']:.4f}  "
          f"CV={cv_score.mean():.4f}±{cv_score.std():.4f}")

# ══════════════════════════════════════════════════════════════
# STEP 5 ── METRICS SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n[5/6] Metrics Summary")
print(f"\n{'Model':<24} {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weighted':>13} {'CV±std':>14}")
print("─" * 76)
for name, r in results.items():
    print(f"{name:<24} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
          f"{r['f1_weighted']:>13.4f} "
          f"{r['cv_mean']:>8.4f}±{r['cv_std']:.4f}")

best_name = max(results, key=lambda k: results[k]['accuracy'])
print(f"\n★  Best Model : {best_name}  (Accuracy = {results[best_name]['accuracy']:.4f})\n")

# Print report for best model (digits + first 5 letters only for brevity)
sample_classes = DIGITS + list('ABCDE')
sample_idx = [i for i, l in enumerate(y_labels) if l in sample_classes]
if len(sample_idx) > 0:
    print(f"Classification Report — {best_name} (digits + A-E):\n")
    mask = np.isin(le.inverse_transform(y_te), sample_classes)
    if mask.sum() > 0:
        print(classification_report(
            y_te[mask],
            results[best_name]['y_pred'][mask],
            target_names=[c for c in le.classes_ if c in sample_classes],
            zero_division=0
        ))

# ══════════════════════════════════════════════════════════════
# STEP 6 ── VISUALISATION DASHBOARD  (9 panels)
# ══════════════════════════════════════════════════════════════
print("[6/6] Generating Dashboard …")

BG           = '#070C18'
CARD         = '#0D1526'
TEXT         = '#F0F4FF'
MUTED        = '#5A6A8A'
MODEL_COLORS = ['#3B82F6', '#10B981', '#F59E0B']
ACCENT       = ['#6366F1','#EC4899','#14B8A6','#F97316','#8B5CF6']

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': CARD,
    'text.color': TEXT,      'axes.labelcolor': TEXT,
    'xtick.color': MUTED,    'ytick.color': MUTED,
    'axes.edgecolor': '#1E2D4A', 'grid.color': '#1E2D4A',
    'font.family': 'DejaVu Sans',
})

fig = plt.figure(figsize=(24, 20))
fig.suptitle('HANDWRITTEN CHARACTER RECOGNITION — EVALUATION DASHBOARD',
             fontsize=17, fontweight='bold', color=TEXT, y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

# ── A: Sample Character Images ────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(CARD)
showcase = DIGITS + list('ABCDEFGHIJKLMNOP')
n_cols, n_rows = 9, 3
grid_img = np.zeros((n_rows * IMG_SIZE, n_cols * IMG_SIZE))
for idx, ch in enumerate(showcase[:n_rows * n_cols]):
    r, c = divmod(idx, n_cols)
    if ch.isdigit():
        img = generate_digit(ch)
    else:
        img = generate_letter(ch)
    grid_img[r*IMG_SIZE:(r+1)*IMG_SIZE, c*IMG_SIZE:(c+1)*IMG_SIZE] = img

ax1.imshow(grid_img, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax1.set_title('Sample Character Images\n(digits 0-9 + letters A-P)',
              color=TEXT, fontweight='bold')
ax1.set_xticks([]); ax1.set_yticks([])

# ── B: Model Accuracy Comparison ─────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
names  = list(results.keys())
accs   = [results[n]['accuracy']  for n in names]
f1s    = [results[n]['f1_macro']  for n in names]
x = np.arange(len(names)); w = 0.35
b1 = ax2.bar(x - w/2, accs, w, color=MODEL_COLORS, alpha=0.90, label='Accuracy')
b2 = ax2.bar(x + w/2, f1s,  w, color=MODEL_COLORS, alpha=0.45, label='F1-Macro')
ax2.set_xticks(x)
ax2.set_xticklabels(['CNN\nMLP', 'SVM\nRBF', 'Random\nForest'], fontsize=9)
ax2.set_ylim(0, 1.12)
ax2.set_title('Model Accuracy vs F1-Macro', color=TEXT, fontweight='bold')
ax2.legend(fontsize=8, framealpha=0.2)
ax2.grid(axis='y', alpha=0.12)
for b, v in zip(b1, accs):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.01,
             f'{v:.3f}', ha='center', fontsize=9, color=TEXT, fontweight='bold')

# ── C: 5-Fold CV Distribution ─────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
cv_data = [cross_val_score(results[n]['model'], X_tr, y_tr,
                            cv=StratifiedKFold(3), scoring='accuracy')
           for n in names]
bp = ax3.boxplot(cv_data, patch_artist=True,
                  medianprops=dict(color='white', lw=2.5))
for patch, c in zip(bp['boxes'], MODEL_COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax3.set_xticklabels(['CNN\nMLP', 'SVM\nRBF', 'Random\nForest'], fontsize=9)
ax3.set_title('3-Fold CV Accuracy', color=TEXT, fontweight='bold')
ax3.set_ylabel('Accuracy'); ax3.set_ylim(0.3, 1.05)
ax3.grid(axis='y', alpha=0.12)

# ── D-F: Confusion Matrices (digits only for clarity) ─────
digit_indices = np.array([i for i, l in enumerate(le.classes_) if l in DIGITS])
digit_names   = [c for c in le.classes_ if c in DIGITS]
mask_te       = np.isin(y_te, digit_indices)

for idx, (name, r) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, idx])
    if mask_te.sum() > 0:
        cm = confusion_matrix(y_te[mask_te], r['y_pred'][mask_te],
                              labels=digit_indices)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                    cmap=sns.light_palette(MODEL_COLORS[idx], as_cmap=True),
                    xticklabels=digit_names, yticklabels=digit_names,
                    linewidths=0.3, linecolor=BG,
                    annot_kws={'size': 8})
    short = name.split()[0]
    ax.set_title(f'Confusion Matrix (digits)\n{short}',
                 color=TEXT, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('Actual',    fontsize=8)
    ax.tick_params(labelsize=8)

# ── G: Per-class Accuracy Bar (best model) ────────────────
ax7 = fig.add_subplot(gs[2, 0])
best_pred = results[best_name]['y_pred']
per_class_acc = []
for i, cls in enumerate(le.classes_):
    mask = (y_te == i)
    if mask.sum() > 0:
        per_class_acc.append(accuracy_score(y_te[mask], best_pred[mask]))
    else:
        per_class_acc.append(0.0)

colors_cls = [MODEL_COLORS[0] if c.isdigit() else MODEL_COLORS[1]
              for c in le.classes_]
ax7.bar(range(len(le.classes_)), per_class_acc,
        color=colors_cls, alpha=0.85)
ax7.set_xticks(range(len(le.classes_)))
ax7.set_xticklabels(list(le.classes_), fontsize=6)
ax7.set_ylim(0, 1.1)
ax7.set_title(f'Per-Class Accuracy ({best_name.split()[0]})\n'
              f'Blue=Digits  Green=Letters',
              color=TEXT, fontweight='bold')
ax7.set_ylabel('Accuracy'); ax7.grid(axis='y', alpha=0.12)
ax7.axhline(np.mean(per_class_acc), color='#F59E0B',
            lw=1.5, ls='--', label=f'Mean={np.mean(per_class_acc):.3f}')
ax7.legend(fontsize=8, framealpha=0.2)

# ── H: PCA Scatter (first 2 components, digits only) ─────
ax8 = fig.add_subplot(gs[2, 1])
digit_mask_tr = np.isin(y_tr, digit_indices)
X_plot  = X_tr[digit_mask_tr][:500, :2]
y_plot  = y_tr[digit_mask_tr][:500]
scatter_colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i, di in enumerate(digit_indices[:10]):
    mask_d = (y_plot == di)
    if mask_d.sum() > 0:
        ax8.scatter(X_plot[mask_d, 0], X_plot[mask_d, 1],
                    c=[scatter_colors[i]], alpha=0.6, s=15,
                    label=le.classes_[di])
ax8.set_title('PCA Feature Space\n(digits, 2 components)',
              color=TEXT, fontweight='bold')
ax8.set_xlabel('PC 1'); ax8.set_ylabel('PC 2')
ax8.legend(fontsize=7, framealpha=0.15, ncol=2)
ax8.grid(alpha=0.12)

# ── I: Augmented Samples Grid ─────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
show_chars = list('0A1B2C3D')
aug_grid   = np.zeros((2 * IMG_SIZE, 4 * IMG_SIZE))
for idx, ch in enumerate(show_chars):
    r, c = divmod(idx, 4)
    base = generate_digit(ch) if ch.isdigit() else generate_letter(ch)
    aug  = augment_image(base)
    aug_grid[r*IMG_SIZE:(r+1)*IMG_SIZE, c*IMG_SIZE:(c+1)*IMG_SIZE] = aug

ax9.imshow(aug_grid, cmap='plasma', aspect='auto', vmin=0, vmax=1)
ax9.set_title('Augmented Training Samples\n(rotation, blur, noise added)',
              color=TEXT, fontweight='bold')
ax9.set_xticks(np.arange(4) * IMG_SIZE + IMG_SIZE//2)
ax9.set_xticklabels(show_chars[::2], fontsize=9, color=TEXT)
ax9.set_yticks([IMG_SIZE//2, IMG_SIZE + IMG_SIZE//2])
ax9.set_yticklabels(['Row 1', 'Row 2'], fontsize=8, color=MUTED)

# Save
plt.tight_layout(rect=[0, 0, 1, 0.97])
dashboard_path = os.path.join(OUTPUT_DIR, 'handwritten_recognition_dashboard.png')
plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"   ✔ Dashboard saved → {dashboard_path}")
plt.close()

# ══════════════════════════════════════════════════════════════
# BONUS: PREDICT FUNCTION  (plug in real image)
# ══════════════════════════════════════════════════════════════
def predict_character(image_path: str) -> str:
    """
    Predict a handwritten character from an image file.

    Requirements:
        pip install Pillow

    Args:
        image_path : path to a 28×28 greyscale image (.png / .jpg)

    Returns:
        Predicted character (digit or letter)

    Usage:
        result = predict_character('my_letter.png')
        print(f'Predicted: {result}')
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert('L').resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0

        # Extract same features as training
        grad   = compute_gradient_features(arr[np.newaxis])[0]
        pixels = arr.flatten()
        combined = np.hstack([pixels, grad])
        scaled   = scaler_pca.transform([combined])
        pca_feat = pca.transform(scaled)

        best_model = results[best_name]['model']
        pred_idx   = best_model.predict(pca_feat)[0]
        return le.inverse_transform([pred_idx])[0]

    except ImportError:
        return "Install Pillow first:  pip install Pillow"
    except Exception as e:
        return f"Error: {e}"

# ── Usage example (uncomment to test) ─────────────────────
# char = predict_character('my_handwritten_A.png')
# print(f'Predicted character: {char}')

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  PROJECT COMPLETE")
print(f"  Best Model  : {best_name}")
print(f"  Accuracy    : {results[best_name]['accuracy']:.4f}")
print(f"  F1-Macro    : {results[best_name]['f1_macro']:.4f}")
print(f"  CV Score    : {results[best_name]['cv_mean']:.4f} ± {results[best_name]['cv_std']:.4f}")
print(f"  Classes     : {len(ALL_CLASSES)} (digits 0-9 + letters A-Z)")
print("=" * 65)
print("\n  Files saved in the same folder as this script:")
print("  📊 handwritten_recognition_dashboard.png")
print("\n  To predict from a real image:")
print("  1. pip install Pillow")
print("  2. Uncomment predict_character() at the bottom")
print("=" * 65)
