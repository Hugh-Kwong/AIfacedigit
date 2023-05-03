import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, num_epochs=100):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(49, 10)
        

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
    #weights will change to an array of 49 weights corresponding to each 4x4 split array look at comments on predict
    def __init__(self, input_size = 27, learning_rate=0.1, num_epochs=1000):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros((10, 49))
    #given an input array we need to divide each array 28x28 into 49 4x4 arrays, make sure that the array is a numpy array so you can use dot product function.
    #this function should take each of these 49 4x4  arrays and multiply them with their corresponding weights and then return the dot product. This will return the value of f(x)
    #ok new update, in digit perceptron, every number 0-9 will have an asssociated perceptron weight vector. We need to run the image through each of the vectors and then choose the highest value to predict
    #then we return the results, weight updating will occur in training not in predict
    def predict(self, inputs):
        splitArr = splitArray(inputs)
        #now we need to calculate the f(x) value for each 0-9 digit perceptrons and select the best result, in range 10 as it goes from 0-9
        #the weights[0][0] corresponds with the 0th perceptron's weights at the 0th weight and splitArr[0][0] refers to the 0,0 vector of the split array
        #not working
        epic = []
        for h in range(10):
            dot = []
            for i in range(6):
                for j in range(6):
                    dot.append(sum(np.dot(self.weights[i][j],splitArr[i][j])))    
            epic.append(sum(dot))
        temp2 = np.array(epic)
        return(temp2.argmax())
             
            

    #to train we run through the array of images for epoch amount of iterations and call prediction on each iteration. The weights will be changed depend on the error given by the prediction.
    #we can count the number of arrays with len(array)
    #self is self, training data is the name of the data file, training labels is file of corresponding labels.
    def train(self, training_data, training_labels):
        temp = digit28Array(training_data)
        ltemp = digitlabelArray(training_labels)
        #i is each array, 
        for i in range(len(temp) ):
            prediction = self.predict(temp[i])
            # print(prediction, ltemp[i])
            split = splitArray(temp[i])
            # print(split[0][0])
            if prediction !=  ltemp[i]:
                count = 0
                print("EPIC STYLE", prediction, ltemp[i])
                for j in range(7):
                    for k in range(7):
                        self.weights[prediction][count] -= sum(split[j][k])
                        print(self.weights[prediction][count])
                        count += 1
            
                
                
                    
        #             self.weights[prediction][j] -= sum(split[prediction][j])
        #             self.weights[ltemp[i]][j] += sum(split[ltemp[i]][j])
        # print(self.weights)
        
            
    #reads in a digit image file and returns a list of 28x28 matrices with " "= 0, # = 1, and + = 2 WE CAN CHANGE THE + TO 1 AS WELL IF LAZY (this will matter much more in the prediction function,  this may change the %success rate)
    #this is new btw much better that I read the instructions a 15th time
   
def splitArray(inputs):
    #self  will refer to the stuff in init inputs will be the the given 28x28 array
    tempArr = np.array(inputs)
    #split the array into 7 4x2 arrays
    splitArr = np.hsplit(tempArr, 7)
    #split the array into 49 4x4 
    for i in range(7):
        splitArr[i] = np.array_split(splitArr[i], 7)
    #transforms each 4x4 into a 16 1D array so ending up with 49 length 16 vectors for the weights
    for i in range(7):
        for j in range(7):
            splitArr[i][j] = splitArr[i][j].flatten()
    return splitArr                   
            
    
        
    
#reads in a digit image file and returns a list of 28x28 matrices with " "= 0, # = 1, and + = 2 WE CAN CHANGE THE + TO 1 AS WELL IF LAZY (this will matter much more in the prediction function,  this may change the %success rate)
#this is new btw much better that I read the instructions a 15th time
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
            current_segment.append([1 if ch == "#" else 2 if ch == "+" else 0 for ch in line.rstrip("\n")])
        if current_segment:
            segments.append(current_segment)        
    return segments


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
#transposes an array, doesnt work very well rn /
#MIGHT NOT NEED
def transposeArray(arr):
    returnList = []
    
    for i in range(len(arr)):
        x = np.array(arr[i])
        x = x.transpose(1, 0)
        returnList.append(x)
    return returnList            
# digits =printArray(digitArray("numbers.txt"),(digitlabelArray("numlab.txt"))   )
# X = digitArray("numbers.txt")

# X =transposeArray(X)
# Y = digitlabelArray("numlab.txt")
X = digit28Array("numbers.txt")
L = np.zeros((10, 49))
L = L
Y = np.array(X[0])
Y = np.hsplit(Y, 7)

# for i in range(len(X)):
#     print(i)
# print(L[0])
M = splitArray(X[0])
# print(M[0][1])
digitP = SelfPerceptron()
digitP.train("numbers.txt", "numlab.txt")
