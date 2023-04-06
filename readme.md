# CSE404 Group Proejcts

- To download data, you can download from release or kaggle page.
    - `data.rar` includes the math symbols
    - `archieve.zip` includes the English

```
├── CNN.py
├── main.py
├── readme.md
├── Data/
│   ├── english.csv
│   ├── Img/
│   └── math/
│       ├── extracted_images
│       └── extracted_images-1
└── 
```

# Model Attempts
## Convolutional Neural Network
```
C:\Users\me\AppData\Local\Programs\Python\Python311\python.exe C:\Users\me\StudioProjects\CSE404Project\NewCNN.py 
                image label
0  Img/img045-049.png     i
1  Img/img052-015.png     p
2  Img/img040-031.png     d
3  Img/img017-014.png     G
4  Img/img030-032.png     T
Found 2910 validated image filenames belonging to 62 classes.
Found 495 validated image filenames belonging to 62 classes.
Found 5 validated image filenames belonging to 5 classes.
Epoch 1/10
91/91 [==============================] - 90s 973ms/step - loss: 3.6888 - accuracy: 0.1354 - val_loss: 2.5964 - val_accuracy: 0.3394
Epoch 2/10
91/91 [==============================] - 86s 949ms/step - loss: 2.0399 - accuracy: 0.4505 - val_loss: 1.8729 - val_accuracy: 0.5152
Epoch 3/10
91/91 [==============================] - 112s 1s/step - loss: 1.2252 - accuracy: 0.6615 - val_loss: 1.6432 - val_accuracy: 0.6061
Epoch 4/10
91/91 [==============================] - 114s 1s/step - loss: 0.8317 - accuracy: 0.7660 - val_loss: 1.7962 - val_accuracy: 0.5939
Epoch 5/10
91/91 [==============================] - 86s 942ms/step - loss: 0.5354 - accuracy: 0.8409 - val_loss: 1.8183 - val_accuracy: 0.5879
Epoch 6/10
91/91 [==============================] - 86s 942ms/step - loss: 0.3967 - accuracy: 0.8835 - val_loss: 1.9269 - val_accuracy: 0.6020
Epoch 7/10
91/91 [==============================] - 83s 912ms/step - loss: 0.3112 - accuracy: 0.9096 - val_loss: 1.7284 - val_accuracy: 0.6525
Epoch 8/10
91/91 [==============================] - 84s 919ms/step - loss: 0.2436 - accuracy: 0.9258 - val_loss: 1.9891 - val_accuracy: 0.6182
Epoch 9/10
91/91 [==============================] - 84s 926ms/step - loss: 0.2044 - accuracy: 0.9381 - val_loss: 1.7882 - val_accuracy: 0.6606
Epoch 10/10
91/91 [==============================] - 82s 901ms/step - loss: 0.1393 - accuracy: 0.9636 - val_loss: 2.0704 - val_accuracy: 0.6505
Prediction mapping:  {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45, 'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55, 'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}
1/1 [==============================] - 0s 383ms/step
Real: i  Pred:  j
Real: p  Pred:  p
Real: d  Pred:  d
Real: G  Pred:  G
Real: T  Pred:  T
```
```
PS C:\Users\muiph\OneDrive\Documents\CSE404Project> python trainCON2.py
Shape of X [N, C, H, W]: torch.Size([32, 3, 900, 1200])
Shape of y: torch.Size([32]) torch.int64
Using cpu device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (conv_relu_stack): Sequential(
    (0): MaxPool2d(kernel_size=16, stride=16, padding=0, dilation=1, ceil_mode=False)
    (1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(512, 62, kernel_size=(3, 3), stride=(1, 1))
  )
  (linear): Linear(in_features=1860, out_features=62, bias=True)
)
Epoch 1
-------------------------------
loss: 4.129909  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.125794

Epoch 2
-------------------------------
loss: 4.125782  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.124796

Epoch 3
-------------------------------
loss: 4.121610  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.124934

Epoch 4
-------------------------------
loss: 4.117387  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.126634

Epoch 5
-------------------------------
loss: 4.115981  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.127449

Epoch 6
-------------------------------
loss: 4.116341  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.127278

Epoch 7

Epoch 10
-------------------------------
loss: 4.113594  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.121228 

Done!
PS C:\Users\muiph\OneDrive\Documents\CSE404Project> 
```

## ResNet50_2

- Epoch 1/20
69/69 [==============================] - 13s 196ms/step - loss: 2.4123 - accuracy: 0.3740 - val_loss: 4.1282 - val_accuracy: 0.0073
- Epoch 2/20
69/69 [==============================] - 9s 136ms/step - loss: 1.0163 - accuracy: 0.6971 - val_loss: 4.1277 - val_accuracy: 0.0220
- Epoch 3/20
69/69 [==============================] - 9s 136ms/step - loss: 0.6895 - accuracy: 0.7809 - val_loss: 4.1276 - val_accuracy: 0.0201
- Epoch 4/20
69/69 [==============================] - 9s 136ms/step - loss: 0.5167 - accuracy: 0.8272 - val_loss: 4.1276 - val_accuracy: 0.0220
- Epoch 5/20
69/69 [==============================] - 9s 137ms/step - loss: 0.4152 - accuracy: 0.8579 - val_loss: 4.1277 - val_accuracy: 0.0201
- Epoch 6/20
69/69 [==============================] - 9s 137ms/step - loss: 0.3764 - accuracy: 0.8616 - val_loss: 4.1277 - val_accuracy: 0.0073
- Epoch 7/20
69/69 [==============================] - 9s 136ms/step - loss: 0.2979 - accuracy: 0.8960 - val_loss: 4.1277 - val_accuracy: 0.0201
- Epoch 8/20
69/69 [==============================] - 9s 137ms/step - loss: 0.2994 - accuracy: 0.8941 - val_loss: 4.1277 - val_accuracy: 0.0201
- Epoch 9/20
69/69 [==============================] - 9s 137ms/step - loss: 0.2654 - accuracy: 0.9070 - val_loss: 4.1277 - val_accuracy: 0.0201
- Epoch 10/20
69/69 [==============================] - 9s 136ms/step - loss: 0.1742 - accuracy: 0.9349 - val_loss: 4.1262 - val_accuracy: 0.0201
- Epoch 11/20
69/69 [==============================] - 9s 136ms/step - loss: 0.2485 - accuracy: 0.9203 - val_loss: 3.8424 - val_accuracy: 0.1337
- Epoch 12/20
69/69 [==============================] - 9s 136ms/step - loss: 0.1973 - accuracy: 0.9290 - val_loss: 2.9942 - val_accuracy: 0.3114
- Epoch 13/20
69/69 [==============================] - 9s 136ms/step - loss: 0.1713 - accuracy: 0.9459 - val_loss: 1.7513 - val_accuracy: 0.5659
- Epoch 14/20
69/69 [==============================] - 9s 136ms/step - loss: 0.1986 - accuracy: 0.9345 - val_loss: 1.2526 - val_accuracy: 0.6319
- Epoch 15/20
69/69 [==============================] - 9s 137ms/step - loss: 0.1955 - accuracy: 0.9363 - val_loss: 0.7002 - val_accuracy: 0.8040
- Epoch 16/20
69/69 [==============================] - 9s 137ms/step - loss: 0.1098 - accuracy: 0.9597 - val_loss: 0.6716 - val_accuracy: 0.7875
- Epoch 17/20
69/69 [==============================] - 9s 136ms/step - loss: 0.0499 - accuracy: 0.9863 - val_loss: 0.6050 - val_accuracy: 0.8077
- Epoch 18/20
69/69 [==============================] - 9s 136ms/step - loss: 0.0404 - accuracy: 0.9895 - val_loss: 0.6964 - val_accuracy: 0.8022
- Epoch 19/20
69/69 [==============================] - 9s 136ms/step - loss: 0.0174 - accuracy: 0.9963 - val_loss: 0.5624 - val_accuracy: 0.8388
- Epoch 20/20
69/69 [==============================] - 9s 136ms/step - loss: 0.0591 - accuracy: 0.9821 - val_loss: 0.7317 - val_accuracy: 0.8114

- more info about this model here [ResNet50_2 more info](https://github.com/RakerZh/character_recogonition)

