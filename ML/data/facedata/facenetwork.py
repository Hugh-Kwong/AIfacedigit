import tensorflow as tf
import numpy as np

def faceArray(filename):
    segment_size = 70
    segments = []
    with open(filename, 'r') as f:
        current_segment = []
        for i, line in enumerate(f):
            segment_num = i // segment_size
            if i % segment_size == 0:
                if current_segment:
                    segments.append(current_segment)
                current_segment = []
            current_segment.append([float(1) if ch == "#" else float(0) for ch in line.rstrip("\n")])
        if current_segment:
            segments.append(current_segment)        
    return segments


#reads in a file containing the labels of the corresponding images [1,2,...]
def facelabelArray(filename):
    with open(filename) as file:
        lines = file.readlines()
    label_arr = []
    for line in lines:
        label_arr.append(int(line.rstrip("\n")))
    return label_arr    

# Load the MNIST dataset
X = faceArray("facedatatrain.txt")
Y = facelabelArray("facedatatrainlabels.txt")
X = np.array(X)
Y = np.array(Y)
test = faceArray("facedatatest.txt")
ltest = facelabelArray("facedatatestlabels.txt") 
test = np.array(test)
ltest = np.array(ltest)
print(len(X))
print(len(Y))
# Convert labels to binary
Y = np.where(Y >= 0.5, 1, 0)
ltest = np.where(ltest >= 0.5, 1, 0)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(70, 60)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = False)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
for i in range(10):
    # Train the model
    indexX = int(len(X) * i+1/10)
    indexY = int(len(Y) * i+1/10)
    pX = X[:indexX]
    pY = X[:indexY]
    # Train the model
    model.fit(X, Y, epochs=5)

    # Evaluate the model
    model.evaluate(test, ltest, verbose=2)
