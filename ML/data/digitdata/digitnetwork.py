import tensorflow as tf
import numpy as np

def digit28Array(filename):
    segment_size = 28
    segments = []
    with open(filename, 'r') as f:
        current_segment = []
        for i, line in enumerate(f):
            segment_num = i // segment_size
            if i % segment_size == 0:
                if current_segment:
                    segments.append(current_segment)
                current_segment = []
            current_segment.append([float(1) if ch == "#" else float(2) if ch == "+" else float(0) for ch in line.rstrip("\n")])
        if current_segment:
            segments.append(current_segment)        
    return segments


#reads in a file containing the labels of the corresponding images [1,2,...]
def digitlabelArray(filename):
    with open(filename) as file:
        lines = file.readlines()
    label_arr = []
    for line in lines:
        label_arr.append(float(line.rstrip("\n")))
    return label_arr    
# Load the MNIST dataset
X = digit28Array("trainingimages.txt")
Y = digitlabelArray("traininglabels.txt")
X = np.array(X)
Y = np.array(Y)
test = digit28Array("testimages.txt")
ltest = digitlabelArray("testlabels.txt") 
test = np.array(test)
ltest = np.array(ltest)

X, test = X / 2, test / 2

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Compile the model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

for i in range(10):
    # Train the model
    indexX = int(len(X) * i+1/10)
    indexY = int(len(Y) * i+1/10)
    pX = X[:indexX]
    pY = X[:indexY]
    print("CURRENT:", (i+1)*10, "%")
    model.fit(X, Y, epochs=5)

    # Evaluate the model
    model.evaluate(test, ltest, verbose=2)

