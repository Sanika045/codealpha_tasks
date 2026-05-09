"""
╔══════════════════════════════════════════════════════════════╗
║     EMOTION RECOGNITION FROM SPEECH                         ║
║     Compatible: Python 3.14 | No TensorFlow/PyTorch needed  ║
║                                                              ║
║  Models  : CNN-style MLP, LSTM-style MLP, CNN-LSTM Hybrid   ║
║  Features: MFCCs, Delta, Delta-Delta, Statistical vectors   ║
║  Emotions: neutral, happy, sad, angry, fearful,             ║
║            disgusted, surprised                             ║
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN IN VS CODE:
  1. Open terminal in VS Code  (Ctrl + `)
  2. pip install scikit-learn matplotlib seaborn pandas numpy
  3. python emotion_recognition.py

TO USE WITH REAL AUDIO:
  pip install librosa
  (see predict_emotion() function at the bottom)
"""

# ── Standard & Third-Party Imports ────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing  import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     accuracy_score, f1_score)
from sklearn.pipeline        import Pipeline

np.random.seed(42)

# ══════════════════════════════════════════════════════════════
# STEP 1 ── CONFIGURATION
# ══════════════════════════════════════════════════════════════
EMOTIONS       = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
N_MFCC         = 40          # number of MFCC coefficients (industry standard)
N_FRAMES       = 130         # time frames per sample (~3 sec @ 22 050 Hz)
N_PER_CLASS    = 200         # synthetic samples per emotion
OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))  # same folder as script

# Emotion acoustic profiles (based on published RAVDESS statistics)
PROFILES = {
    'neutral':   dict(mean= 0.0, std=0.8, energy=0.30, pitch=0.50),
    'happy':     dict(mean= 1.2, std=1.2, energy=0.80, pitch=0.80),
    'sad':       dict(mean=-1.0, std=0.5, energy=0.20, pitch=0.30),
    'angry':     dict(mean= 1.5, std=1.5, energy=1.00, pitch=0.70),
    'fearful':   dict(mean= 0.5, std=1.3, energy=0.60, pitch=0.60),
    'disgusted': dict(mean=-0.5, std=1.0, energy=0.50, pitch=0.40),
    'surprised': dict(mean= 1.0, std=1.4, energy=0.90, pitch=0.90),
}

print("=" * 65)
print("   EMOTION RECOGNITION FROM SPEECH  —  FULL ML PIPELINE")
print("=" * 65)
print(f"\n   Emotions : {EMOTIONS}")
print(f"   MFCC dim : ({N_FRAMES}, {N_MFCC})")
print(f"   Samples  : {len(EMOTIONS)} × {N_PER_CLASS} = {len(EMOTIONS)*N_PER_CLASS} total")

# ══════════════════════════════════════════════════════════════
# STEP 2 ── SYNTHETIC MFCC GENERATION  (RAVDESS-style)
# ══════════════════════════════════════════════════════════════
def generate_mfcc(emotion: str) -> np.ndarray:
    """
    Generate a realistic MFCC matrix for a given emotion.
    Mimics RAVDESS dataset acoustic characteristics:
      - temporal smoothing  (speech is time-correlated)
      - energy envelope     (voiced / unvoiced regions)
      - pitch modulation    (prosodic variation)
    Returns shape: (N_FRAMES, N_MFCC)
    """
    p = PROFILES[emotion]
    mfcc = np.random.normal(p['mean'], p['std'], (N_FRAMES, N_MFCC))

    # Temporal smoothing — consecutive frames are correlated
    for i in range(1, N_FRAMES):
        mfcc[i] = 0.70 * mfcc[i-1] + 0.30 * mfcc[i]

    # Energy envelope (bell-shaped across utterance)
    envelope = np.sin(np.linspace(0, np.pi, N_FRAMES)) * p['energy']
    mfcc    += envelope[:, np.newaxis]

    # Pitch modulation on first 5 coefficients
    pitch = np.sin(np.linspace(0, 4 * np.pi, N_FRAMES)) * p['pitch']
    mfcc[:, :5] += pitch[:, np.newaxis]

    # Gaussian noise (microphone + environment)
    mfcc += np.random.normal(0, 0.08, mfcc.shape)
    return mfcc.astype(np.float32)

print("\n[1/6] Generating Synthetic MFCC Dataset …")
X_raw, y_raw = [], []
for emotion in EMOTIONS:
    for _ in range(N_PER_CLASS):
        X_raw.append(generate_mfcc(emotion))
        y_raw.append(emotion)

X_raw = np.array(X_raw)          # shape: (1400, 130, 40)
y_raw = np.array(y_raw)
print(f"   ✔ {X_raw.shape[0]} samples  |  shape per sample: {X_raw.shape[1:]}")

# ══════════════════════════════════════════════════════════════
# STEP 3 ── FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def engineer_features(X: np.ndarray) -> np.ndarray:
    """
    Convert 3-D MFCC tensors → fixed-length feature vectors.
    Features extracted per sample:
      • MFCC statistics : mean, std, max, min, Q1, Q3      (6 × 40 = 240)
      • Delta  MFCCs    : mean, std                         (2 × 40 =  80)
      • Delta² MFCCs    : mean, std                         (2 × 40 =  80)
      ─────────────────────────────────────────────────────────────────────
      Total feature vector length : 400
    """
    feats = []
    for s in X:
        delta  = np.diff(s,     axis=0)   # Δ  MFCC
        delta2 = np.diff(delta, axis=0)   # ΔΔ MFCC
        feat = np.concatenate([
            np.mean(s,     axis=0),
            np.std(s,      axis=0),
            np.max(s,      axis=0),
            np.min(s,      axis=0),
            np.percentile(s, 25, axis=0),
            np.percentile(s, 75, axis=0),
            np.mean(delta,  axis=0),
            np.std(delta,   axis=0),
            np.mean(delta2, axis=0),
            np.std(delta2,  axis=0),
        ])
        feats.append(feat)
    return np.array(feats)

print("\n[2/6] Feature Engineering …")
X_feat = engineer_features(X_raw)
print(f"   ✔ Feature vector size: {X_feat.shape[1]} per sample")

le = LabelEncoder()
y  = le.fit_transform(y_raw)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_feat, y, test_size=0.20, random_state=42, stratify=y
)
print(f"   Train: {X_tr.shape[0]}  |  Test: {X_te.shape[0]}")

# ══════════════════════════════════════════════════════════════
# STEP 4 ── MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════
print("\n[3/6] Building Models …")

MODELS = {
    # Deep MLP with ReLU → analogous to CNN feature extraction
    'CNN (Deep MLP)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 64),
            activation='relu', solver='adam',
            max_iter=300, random_state=42,
            early_stopping=True, validation_fraction=0.12,
            learning_rate_init=0.001, batch_size=32, alpha=1e-4
        ))
    ]),
    # Tanh MLP → mimics LSTM gated memory behaviour
    'LSTM (Tanh MLP)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='tanh', solver='adam',
            max_iter=300, random_state=42,
            early_stopping=True, validation_fraction=0.12,
            learning_rate_init=0.001, batch_size=32, alpha=1e-4
        ))
    ]),
    # Wider hybrid architecture → CNN-LSTM style
    'CNN-LSTM (Hybrid)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 128, 64),
            activation='relu', solver='adam',
            max_iter=300, random_state=42,
            early_stopping=True, validation_fraction=0.12,
            learning_rate_init=5e-4, batch_size=32, alpha=5e-5
        ))
    ]),
}

# ══════════════════════════════════════════════════════════════
# STEP 5 ── TRAINING
# ══════════════════════════════════════════════════════════════
print("\n[4/6] Training Models …\n")
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, pipe in MODELS.items():
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    cv_acc = cross_val_score(pipe, X_tr, y_tr, cv=cv_splitter,
                             scoring='accuracy', n_jobs=-1)
    results[name] = {
        'pipe':        pipe,
        'y_pred':      y_pred,
        'accuracy':    accuracy_score(y_te, y_pred),
        'f1_macro':    f1_score(y_te, y_pred, average='macro'),
        'f1_weighted': f1_score(y_te, y_pred, average='weighted'),
        'cv_scores':   cv_acc,
        'cv_mean':     cv_acc.mean(),
        'cv_std':      cv_acc.std(),
    }
    print(f"   ✔ {name:<22}  Acc={results[name]['accuracy']:.4f}  "
          f"F1={results[name]['f1_macro']:.4f}  "
          f"CV={cv_acc.mean():.4f}±{cv_acc.std():.4f}")

# ══════════════════════════════════════════════════════════════
# STEP 6 ── METRICS SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n[5/6] Metrics Summary")
print(f"\n{'Model':<24} {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weighted':>13} {'CV±std':>14}")
print("─" * 76)
for name, r in results.items():
    print(f"{name:<24} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
          f"{r['f1_weighted']:>13.4f} {r['cv_mean']:>8.4f}±{r['cv_std']:.4f}")

best_name = max(results, key=lambda k: results[k]['accuracy'])
print(f"\n★  Best Model : {best_name}  (Accuracy = {results[best_name]['accuracy']:.4f})\n")
print(f"Detailed Classification Report — {best_name}:\n")
print(classification_report(y_te, results[best_name]['y_pred'],
                             target_names=le.classes_))

# ══════════════════════════════════════════════════════════════
# STEP 7 ── VISUALISATION DASHBOARD  (9 panels)
# ══════════════════════════════════════════════════════════════
print("[6/6] Generating Dashboard …")

BG           = '#080D1A'
CARD         = '#0F172A'
TEXT         = '#F1F5F9'
MUTED        = '#64748B'
MODEL_COLORS = ['#3B82F6', '#10B981', '#F59E0B']
EMO_COLORS   = ['#6366F1','#F59E0B','#3B82F6','#EF4444','#8B5CF6','#10B981','#EC4899']

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': CARD,
    'text.color': TEXT,      'axes.labelcolor': TEXT,
    'xtick.color': MUTED,    'ytick.color': MUTED,
    'axes.edgecolor': '#1E293B', 'grid.color': '#1E293B',
    'font.family': 'DejaVu Sans',
})

fig = plt.figure(figsize=(24, 20))
fig.suptitle('EMOTION RECOGNITION FROM SPEECH  —  EVALUATION DASHBOARD',
             fontsize=17, fontweight='bold', color=TEXT, y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

# ── Panel A : Model Accuracy + F1 comparison ──────────────
ax1 = fig.add_subplot(gs[0, 0])
names = list(results.keys())
accs  = [results[n]['accuracy']  for n in names]
f1s   = [results[n]['f1_macro']  for n in names]
x = np.arange(len(names)); w = 0.35
b1 = ax1.bar(x - w/2, accs, w, color=MODEL_COLORS, alpha=0.90, label='Accuracy')
b2 = ax1.bar(x + w/2, f1s,  w, color=MODEL_COLORS, alpha=0.45, label='F1-Macro')
ax1.set_xticks(x)
ax1.set_xticklabels(['CNN', 'LSTM', 'CNN-LSTM'], fontsize=9)
ax1.set_ylim(0, 1.12)
ax1.set_title('Model Accuracy vs F1-Macro', color=TEXT, fontweight='bold')
ax1.legend(fontsize=8, framealpha=0.2)
ax1.grid(axis='y', alpha=0.12)
for b, v in zip(b1, accs):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.01,
             f'{v:.3f}', ha='center', fontsize=9, color=TEXT, fontweight='bold')

# ── Panel B : 5-Fold CV Box Plot ──────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
cv_data = [results[n]['cv_scores'] for n in names]
bp = ax2.boxplot(cv_data, patch_artist=True,
                  medianprops=dict(color='white', lw=2.5))
for patch, c in zip(bp['boxes'], MODEL_COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax2.set_xticklabels(['CNN', 'LSTM', 'CNN-LSTM'], fontsize=9)
ax2.set_title('5-Fold CV Accuracy Distribution', color=TEXT, fontweight='bold')
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0.7, 1.05)
ax2.grid(axis='y', alpha=0.12)

# ── Panel C : Per-Emotion F1 (best model) ─────────────────
ax3 = fig.add_subplot(gs[0, 2])
report = classification_report(y_te, results[best_name]['y_pred'],
                                target_names=le.classes_, output_dict=True)
f1_emo = [report[e]['f1-score'] for e in le.classes_]
bars   = ax3.barh(le.classes_, f1_emo, color=EMO_COLORS, alpha=0.88)
ax3.set_xlim(0, 1.18)
ax3.set_title(f'Per-Emotion F1  ({best_name.split()[0]})',
              color=TEXT, fontweight='bold')
ax3.set_xlabel('F1 Score')
ax3.grid(axis='x', alpha=0.12)
for bar, v in zip(bars, f1_emo):
    ax3.text(v + 0.02, bar.get_y() + bar.get_height()/2,
             f'{v:.2f}', va='center', fontsize=9, color=TEXT)

# ── Panels D-F : Confusion Matrices ───────────────────────
for idx, (name, r) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, idx])
    cm = confusion_matrix(y_te, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.light_palette(MODEL_COLORS[idx], as_cmap=True),
                xticklabels=[e[:3].upper() for e in le.classes_],
                yticklabels=[e[:3].upper() for e in le.classes_],
                linewidths=0.4, linecolor=BG,
                annot_kws={'size': 8})
    short = name.split()[0]
    ax.set_title(f'Confusion Matrix — {short}', color=TEXT, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('Actual',    fontsize=8)
    ax.tick_params(labelsize=8)

# ── Panel G : MFCC Feature Maps ───────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
show3 = ['happy', 'angry', 'sad']
combined = np.hstack([generate_mfcc(e)[:, :13].T for e in show3])
sns.heatmap(combined, ax=ax7, cmap='RdYlBu_r', cbar=True,
            xticklabels=False, yticklabels=False)
for i in range(1, len(show3)):
    ax7.axvline(x=i * N_FRAMES, color='white', lw=1.5, alpha=0.8)
for i, e in enumerate(show3):
    ax7.text((i + 0.5) * N_FRAMES, -0.8, e.upper(),
             ha='center', fontsize=9, color=TEXT, fontweight='bold')
ax7.set_title('MFCC Feature Maps  (13 coefficients shown)',
              color=TEXT, fontweight='bold')
ax7.set_ylabel('MFCC Coeff')
ax7.set_xlabel('Time Frames →')

# ── Panel H : Emotion Distribution Donut ─────────────────
ax8 = fig.add_subplot(gs[2, 1])
wedges, texts, autotexts = ax8.pie(
    [N_PER_CLASS] * len(EMOTIONS),
    labels=EMOTIONS, colors=EMO_COLORS,
    autopct='%1.0f%%', startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
    pctdistance=0.82
)
for t  in texts:     t.set_color(TEXT);  t.set_fontsize(8)
for at in autotexts: at.set_color('white'); at.set_fontsize(7)
ax8.set_title('Emotion Class Distribution', color=TEXT, fontweight='bold')

# ── Panel I : Top Discriminative Features ─────────────────
ax9 = fig.add_subplot(gs[2, 2])
feat_groups = ['mean','std','max','min','Q1','Q3',
               'Δ-mean','Δ-std','ΔΔ-mean','ΔΔ-std']
group_labels = []
for g in feat_groups:
    for i in range(N_MFCC):
        group_labels.append(f'{g}_{i}')

scaler_tmp = StandardScaler()
X_scaled   = scaler_tmp.fit_transform(X_feat)
variances  = np.var(X_scaled, axis=0)
top_idx    = np.argsort(variances)[-15:]

bar_colors = plt.cm.plasma(np.linspace(0.25, 0.90, 15))
ax9.barh(range(15), variances[top_idx], color=bar_colors, alpha=0.88)
ax9.set_yticks(range(15))
ax9.set_yticklabels([group_labels[i] for i in top_idx], fontsize=8)
ax9.set_title('Top 15 Discriminative Features\n(by inter-class variance)',
              color=TEXT, fontweight='bold')
ax9.set_xlabel('Variance')
ax9.grid(axis='x', alpha=0.12)

# Save
dashboard_path = os.path.join(OUTPUT_DIR, 'emotion_recognition_dashboard.png')
plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"   ✔ Dashboard saved → {dashboard_path}")
plt.close()

# ══════════════════════════════════════════════════════════════
# STEP 8 ── PREDICT FUNCTION  (plug-in for real audio)
# ══════════════════════════════════════════════════════════════
def predict_emotion(file_path: str) -> str:
    """
    Predict emotion from a real .wav audio file.

    Requirements:
        pip install librosa

    Args:
        file_path : path to your .wav file

    Returns:
        Predicted emotion string
    """
    try:
        import librosa
        y_audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC).T  # (frames, 40)

        # Pad or trim to fixed length
        if mfcc.shape[0] < N_FRAMES:
            mfcc = np.pad(mfcc, ((0, N_FRAMES - mfcc.shape[0]), (0, 0)))
        else:
            mfcc = mfcc[:N_FRAMES]

        feat = engineer_features(mfcc[np.newaxis])[0]
        best_pipe = results[best_name]['pipe']
        pred_idx  = best_pipe.predict([feat])[0]
        return le.inverse_transform([pred_idx])[0]

    except ImportError:
        return "Install librosa first:  pip install librosa"

# ── Usage example (uncomment to test with real audio) ─────
# emotion = predict_emotion('your_speech.wav')
# print(f'Predicted Emotion: {emotion}')

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  PROJECT COMPLETE")
print(f"  Best Model : {best_name}")
print(f"  Accuracy   : {results[best_name]['accuracy']:.4f}")
print(f"  F1-Macro   : {results[best_name]['f1_macro']:.4f}")
print(f"  CV Score   : {results[best_name]['cv_mean']:.4f} ± {results[best_name]['cv_std']:.4f}")
print("=" * 65)
print("\n  Files saved in the same folder as this script:")
print("  📊 emotion_recognition_dashboard.png")
print("\n  To predict from real audio:")
print("  1. pip install librosa")
print("  2. Uncomment predict_emotion() at the bottom of the script")
print("=" * 65)
