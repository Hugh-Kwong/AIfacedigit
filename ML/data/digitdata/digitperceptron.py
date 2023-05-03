import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, num_epochs=100):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(input_size + 1)

    def predict(self, inputs):
        inputs = np.hstack((inputs, [1]))
        activation = np.dot(inputs, self.weights)
        return 1 if activation >= 0 else 0

    def train(self, train_data, train_labels):
        for epoch in range(self.num_epochs):
            for inputs, label in zip(train_data, train_labels):
                inputs_flat = np.hstack(inputs)
                prediction = self.predict(inputs_flat)
                error = label - prediction
                self.weights += self.learning_rate * error * np.hstack((inputs_flat, [1]))

    def evaluate(self, test_data, test_labels):
        correct = 0
        for inputs, label in zip(test_data, test_labels):
            inputs_flat = np.hstack(inputs)
            prediction = self.predict(inputs_flat)
            if prediction == label:
                correct += 1
        return correct / len(test_data)
class SelfPerceptron:
    def __init__(self, input_size, learning_rate=0.1, num_epochs=100):
        self.input_size = 27
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(input_size + 1)
    def predict(self, inputs):
        activation = np.dot(inputs, self.weights)
        return 1 if activation >= 0 else 0 
    def train(self, inputs):s
        
    
#reads in a file into a array of the images [[[0,... 1],[ 0,.. 1],...], [[0,... 1],[ 0,.. 1],...],....]
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
#reads in a file containing the labels of the corresponding images [1,2,...]
def digitlabelArray(filename):
    with open(filename) as file:
        lines = file.readlines()
    label_arr = []
    for line in lines:
        label_arr.append(line.rstrip("\n"))
    return label_arr    


def printArray(arr):
    
    for i in range(len(arr)):
        print()
        for j in range(len(arr[i])):
            print(arr[i][j])
#transposes an array, doesnt work very well rn
def transposeArray(arr):
    returnList = []
    
    for i in range(len(arr)):
        x = np.array(arr[i])
        x = x.transpose(1, 0)
        returnList.append(x)
    return returnList            
# digits =printArray(digitArray("numbers.txt"),(digitlabelArray("numlab.txt"))   )
X = digitArray("numbers.txt")

X =transposeArray(X)
Y = digitlabelArray("numlab.txt")

printArray(X)
