"""Train the cat-only (12 breed) model on Oxford-IIIT Pet.

    python train_cats.py                              # MobileNetV2, ~10 MB
    python train_cats.py --backbone efficientnetv2b0  # stronger, ~25 MB (use the GPU)

Saves cat_breed_model_v2.keras + cat_class_names.txt next to this file;
main.py and predict.py pick them up automatically.
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_loader import load_cat_dataset
from model_builder import BACKBONES, build_cat_model, unfreeze_top
from utils import load_class_names, is_cat_breed

BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", choices=BACKBONES, default="mobilenetv2")
parser.add_argument("--epochs-head", type=int, default=10, help="stage 1: train the new head only")
parser.add_argument("--epochs-ft", type=int, default=20, help="stage 2: fine-tune the top of the backbone")
parser.add_argument("--unfreeze", type=int, default=40, help="backbone layers to unfreeze in stage 2")
parser.add_argument("--label-smoothing", type=float, default=0.1)
parser.add_argument("--mixed-precision", action="store_true",
                    help="float16 on the GPU: faster, but the saved model then runs slowly on a CPU")
parser.add_argument("--out-dir", type=Path, default=BASE_DIR, help="where to save the model + class names")
args = parser.parse_args()

tf.keras.utils.set_random_seed(42)
gpus = tf.config.list_physical_devices("GPU")
print(f"GPU: {[g.name for g in gpus] or 'none - training on CPU (slow)'}")
if args.mixed_precision and gpus:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

train, val, test, class_names, n_train = load_cat_dataset()
num_classes = len(class_names)
steps_per_epoch = int(np.ceil(n_train / 32))
print(f"{num_classes} breeds, {n_train} training images")

def one_hot(images, labels):
    return images, tf.one_hot(labels, num_classes)

train_oh, val_oh = train.map(one_hot), val.map(one_hot)

def compile_model(model, lr, epochs):
    # Cosine decay from lr to ~0 across this stage
    schedule = tf.keras.optimizers.schedules.CosineDecay(lr, max(1, epochs * steps_per_epoch))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(3, name="top3")],
    )

def callbacks():
    return [tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=4, restore_best_weights=True)]

model, base = build_cat_model(num_classes, args.backbone)

print("\n== Stage 1: training the head (backbone frozen)")
compile_model(model, 1e-3, args.epochs_head)
model.fit(train_oh, validation_data=val_oh, epochs=args.epochs_head, callbacks=callbacks())

if args.epochs_ft > 0:
    print(f"\n== Stage 2: fine-tuning the top {args.unfreeze} backbone layers")
    unfreeze_top(base, args.unfreeze)
    compile_model(model, 1e-5, args.epochs_ft)
    model.fit(train_oh, validation_data=val_oh, epochs=args.epochs_ft, callbacks=callbacks())

# ── Save ─────────────────────────────────────────────────────
args.out_dir.mkdir(parents=True, exist_ok=True)
model.save(args.out_dir / "cat_breed_model_v2.keras")
(args.out_dir / "cat_class_names.txt").write_text("\n".join(class_names) + "\n")
print(f"\nSaved {args.out_dir / 'cat_breed_model_v2.keras'}")

# ── Report on the held-out test set ──────────────────────────
y_true = np.concatenate([y.numpy() for _, y in test])
probs = model.predict(test, verbose=0)
y_pred = probs.argmax(axis=1)
top3 = np.mean([t in row for t, row in zip(y_true, np.argsort(probs, axis=1)[:, -3:])])
acc = np.mean(y_pred == y_true)

# Baseline: the original 37-class model, probabilities kept to the 12 cat classes
old_names = load_class_names()
old_cat_ids = [i for i, n in enumerate(old_names) if is_cat_breed(n)]
old_model = tf.keras.models.load_model(BASE_DIR / "cat_breed_model.h5")
old_probs = np.concatenate([old_model.predict(x.numpy() / 255.0, verbose=0)[:, old_cat_ids] for x, _ in test])
old_acc = np.mean(old_probs.argmax(axis=1) == y_true)

print(f"\n== Test set ({len(y_true)} cat images)")
print(f"   old model   top-1: {old_acc:.1%}")
print(f"   new model   top-1: {acc:.1%}   top-3: {top3:.1%}")

print("\n== Per-breed accuracy (and the breed it is most confused with)")
confusion = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    confusion[t, p] += 1
for i, name in enumerate(class_names):
    row = confusion[i].copy()
    correct = row[i]
    row[i] = 0
    worst = f"  <- often {class_names[row.argmax()]} ({row.max()})" if row.max() > 0 else ""
    print(f"   {name:18s} {correct / confusion[i].sum():6.1%}{worst}")
