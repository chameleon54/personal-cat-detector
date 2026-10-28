import tensorflow as tf

def build_model(num_classes=37, img_size=224):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

BACKBONES = ("mobilenetv2", "efficientnetv2b0")

def build_cat_model(num_classes=12, backbone="mobilenetv2", img_size=224, dropout=0.3):
    """Cat-only classifier. Takes raw 0..255 images: augmentation (training only)
    and the backbone's own preprocessing live inside the model.
    Returns (model, base) - keep `base` to unfreeze it for fine-tuning.
    """
    shape = (img_size, img_size, 3)
    inputs = tf.keras.Input(shape=shape)

    x = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.2),
    ], name="augmentation")(inputs)

    if backbone == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights='imagenet')
        x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)  # 0..255 -> -1..1
    elif backbone == "efficientnetv2b0":
        # Rescales 0..255 itself (include_preprocessing=True)
        base = tf.keras.applications.EfficientNetV2B0(
            input_shape=shape, include_top=False, weights='imagenet', include_preprocessing=True
        )
    else:
        raise ValueError(f"backbone must be one of {BACKBONES}")
    base.trainable = False

    x = base(x, training=False)  # keep BatchNorm in inference mode, even when fine-tuning
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs, outputs, name=f"cat_{backbone}"), base

def unfreeze_top(base, n_layers=40):
    """Make the top n layers of the backbone trainable (BatchNorm stays frozen)."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
