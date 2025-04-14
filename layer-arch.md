#### Understanding the Input Data and recommended layers

```
tf.keras.sequential()

# Use Conv2D when you want to extract spatial features from images.
# Use Dense layers for decision making/classification after features are extracted.
# Use Flatten when moving from convolutional layers to dense layers.
# For simpler data (like MNIST), sometimes Dense is enough.
```

| Data type | Recommended Layers | Use Case |
|-----------|--------------------|----------|
| 1D data | Dense, LSTM, GRU, Embedding | Text, Time series|
| 2D image (grayscale or RGB) | Conv2D, MaxPooling2D, Flatten, Dense | Images like MNIST, Flowers|
| 3D image/Video | Conv3D, MaxPooling3D | Medical Imaging, Videos|

| Goal/Data type | What to Use |
|----------------|-------------|
|Image Classification | Conv2D, MaxPooling2D, Flatten, Dense |
|Text classification | Embedding, LSTM/GRU, Dense|
|Tabular data| Dense, Dropout|
|Time series| LSTM, GRU, 1D Conv|
|Reduce overfitting | Add Dropout, regularization|
|Scale input pixels| Rescaling, or manually divide by 255|


###### A basic building block for image classification model

```
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(height, width, channels)),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(num_classes)  # No activation if using loss like SparseCategoricalCrossentropy(from_logits=True)
])
```

| Layer	| Purpose |
|-------|---------|
|Conv2D|Extract features like edges, textures, shapes|
|MaxPooling2D|Downsample the image, reduce computation and overfitting|
|Flatten|Convert 2D features into 1D for classification|
|Dense|Make decisions / classifications based on learned features|
|Dropout (optional)|Prevent overfitting by randomly turning off neurons|
|Rescaling (optional)| Normalize pixel values to 0–1|

###### Notes:
Input Shape: (height, width, channels) — e.g. (150, 150, 3) for RGB images.
You can add more Conv2D+MaxPooling2D pairs to extract deeper features.
Dropout is great for preventing overfitting especially with small datasets.
Final Dense(num_classes) layer gives raw scores (logits) unless you add Softmax.