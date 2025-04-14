Data Type	Recommended Layers	Example Use
1D data	Dense, LSTM, GRU, Embedding	Text, time series
2D image (grayscale or RGB)	Conv2D, MaxPooling2D, Flatten, Dense	Images like MNIST, Flowers
3D image / video	Conv3D, MaxPooling3D	Medical imaging, videos


Goal / Data Type	What to Use
Image classification	Conv2D, MaxPooling2D, Flatten, Dense
Text classification	Embedding, LSTM/GRU, Dense
Tabular data	Dense, Dropout
Time series	LSTM, GRU, 1D Conv
Reduce overfitting	Add Dropout, regularization
Scale input pixels	Rescaling, or manually divide by 255

| Data type | Recommended Layers | Use Case |
|-----------|--------------------|----------|
| 1D data | Dense, LSTM, GRU, Embedding | Text, Time series|
| 2D image (grayscale or RGB) | Conv2D, MaxPooling2D, Flatten, Dense | Images like MNIST, Flowers|
| 3D image/Video | Conv3D, MaxPooling3D | Medical Imaging, Videos|