import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf

# Crear una lista con las rutas de archivo para entrenamiento y prueba
train_dir = Path('fruta-verdura/entrenamiento')
train_filepaths = list(train_dir.glob(r'**/*.jpg'))

test_dir = Path('fruta-verdura/test')
test_filepaths = list(test_dir.glob(r'**/*.jpg'))

val_dir = Path('fruta-verdura/validacion')
val_filepaths = list(val_dir.glob(r'**/*.jpg'))


def proc_img(filepath):

    labels = [str(filepath[i]).split("/")[-2]
              for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenar rutas de archivo y etiquetas
    df = pd.concat([filepath, labels], axis=1)

    # Mezclar el DataFrame y resetear el índice
    df = df.sample(frac=1).reset_index(drop=True)

    return df


train_df = proc_img(train_filepaths)
test_df = proc_img(test_filepaths)
val_df = proc_img(val_filepaths)

print('-- Set de entrenamiento --\n')
print(f'Número de imágenes: {train_df.shape[0]}\n')
print(f'Número de etiquetas: {len(train_df.Label.unique())}\n')
print(f'Etiquetas: {train_df.Label.unique()}')

# Crear un DataFrame con una etiqueta de cada categoría
df_unique = train_df.copy().drop_duplicates(subset=["Label"]).reset_index()

# Mostrar algunas imágenes del conjunto de datos
fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(8, 7),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df_unique.Filepath[i]))
    ax.set_title(df_unique.Label[i], fontsize=12)
plt.tight_layout(pad=0.5)
plt.show()

# Generadores de imágenes para entrenamiento, validación y prueba
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_images = train_generator.flow_from_dataframe(
    dataframe=val_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# Cargar el modelo preentrenado
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
pretrained_model.trainable = False

inputs = pretrained_model.input

x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)

outputs = tf.keras.layers.Dense(36, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_images,
    validation_data=val_images,
    batch_size=32,
    epochs=8,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
    ]
)

pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
plt.title("Accuracy")
plt.show()

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.title("Loss")
plt.show()

# Predecir la etiqueta de las imágenes de prueba
pred = model.predict(test_images)
pred = np.argmax(pred, axis=1)

# Mapear la etiqueta
labels = (train_images.class_indices)
labels = dict((v, k) for k, v in labels.items())
pred = [labels[k] for k in pred]

y_test = [labels[k] for k in test_images.classes]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    img = plt.imread(test_df.Filepath.iloc[i])
    ax.imshow(img)
    ax.set_title(f"Nombre real: {test_df.Label.iloc[i]}\nPredicho: {pred[i]}", fontsize=12)
    ax.set_frame_on(False)
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.tight_layout(pad=1.0)
plt.show()

