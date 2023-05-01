import numpy as np

#WARNING LEGIT NONE OF THE PERCEPTRON CLASS WORKS
# Define the perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=50):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)

    # Define the activation function
    def activation(self, x):
        return 1 if x >= 0 else 0

    # Define the predict function
    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.weights.T.dot(x)
        a = self.activation(z)
        return a

    # Define the train function
    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x = X[i]
                y_pred = self.predict(x)
                error = y[i] - y_pred
                x = np.insert(x, 0, 1)
                self.weights += self.learning_rate * error * x
    
    
    
    
def digitArray(filename):
    with open(filename) as file:
        lines = file.readlines()

    num_arr = []
    temp_arr = []
    for line in lines:
        if '+' in line or '#' in line:
            temp_arr.append([1 if ch == "#" else 2 if ch == "+" else 0 for ch in line.rstrip("\n")])
        elif(temp_arr != []):
            num_arr.append(temp_arr)
            temp_arr = []
            continue
    return num_arr

def digitlabelArray(filename):
    with open(filename) as file:
        lines = file.readlines()
    label_arr = []
    for line in lines:
        label_arr.append(line.rstrip("\n"))
    return label_arr    


def printArray(arr, larr):
    for i in range(len(arr)):
        print()
        print(larr[i])
        for j in range(len(arr[i])):
            print(arr[i][j])

        
digits = digitArray("trainingimages.txt")
dlabels = digitlabelArray("traininglabels.txt")            
# Create a perceptron with input size 784 (since each image is 28x28)
perceptron = Perceptron(input_size=784)

# Train the perceptron on the data
perceptron.train(digits, dlabels)

# Predict the label of a new image
new_image = [0,0,0,0,0,0,0,0,0,0,2,1,1,1,1,1,1,1,1,1,2,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,2,1,1,1,1,1,1,1,1,1,1,1,2,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,1,1,1,1,1,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,2,1,1,2,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,2,1,1,1,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,2,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,2,2,1,1,1,2,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,1,0,0,0,0,0,0,
0,0,0,0,0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
0,0,0,0,0,2,1,1,1,2,2,0,0,0,0,0,0,0,2,1,1,1,0,0,0,0,0,0,
0,0,0,0,0,0,1,1,1,1,1,1,2,2,0,0,0,2,1,1,1,2,0,0,0,0,0,0,
0,0,0,0,0,0,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,2,2,1,1,1,1,1,1,1,1,1,1,1,2,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,2,1,1,1,1,1,1,1,1,1,2,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,2,2,1,1,1,1,2,2,0,0,0,0,0,0,0,0,0,]
  # 28x28 flattened into a 784-dimensional vector
predicted_label = perceptron.predict(new_image)
