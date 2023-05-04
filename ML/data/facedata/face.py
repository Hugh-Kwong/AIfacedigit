import numpy as np


class SelfPerceptron:
    #weights will change to an array of 49 weights corresponding to each 4x4 split array look at comments on predict
    def __init__(self):
        self.weights = np.zeros(100)
    #given an input array we need to divide each array 28x28 into 49 4x4 arrays, make sure that the array is a numpy array so you can use dot product function.
    #this function should take each of these 49 4x4  arrays and multiply them with their corresponding weights and then return the dot product. This will return the value of f(x)
    #ok new update, in digit perceptron, every number 0-9 will have an asssociated perceptron weight vector. We need to run the image through each of the vectors and then choose the highest value to predict
    #then we return the results, weight updating will occur in training not in predict
    def predict(self, inputs):
        splitArr = splitArray(inputs)
        #now we need to calculate the f(x) value for each 0-9 digit perceptrons and select the best result, in range 10 as it goes from 0-9
        #the weights[0][0] corresponds with the 0th perceptron's weights at the 0th weight and splitArr[0][0] refers to the 0,0 vector of the split array
        #not working
        epic = 0
        count = 0
        for i in range(10):
            for j in range(10):
                epic += sum(np.dot(self.weights[count],splitArr[i][j]))  
                count += 1
            # epic.append(sum(dot)
        return epic
             
            

    #to train we run through the array of images for epoch amount of iterations and call prediction on each iteration. The weights will be changed depend on the error given by the prediction.
    #we can count the number of arrays with len(array)
    #self is self, training data is the name of the data file, training labels is file of corresponding labels.
    def train(self, training_data, training_labels):
        temp = faceArray(training_data)
        ltemp = facelabelArray(training_labels)
        #i is each array, 
        for z in range(10):
            for i in range(len(temp)):
                prediction = self.predict(temp[i])
                split = splitArray(temp[i])
                prediction = int(prediction)
                label = int(ltemp[i])
                if prediction >= 0 and label == 0:
                    count = 0
                    for j in range(10):
                        for k in range(10):
                            self.weights[count] -= sum(split[j][k])
                            count+= 1
                elif prediction < 0 and label == 1:
                    count = 0
                    for j in range(10):
                        for k in range(10):
                            self.weights[count] += sum(split[j][k])
                            count += 1
                    
        # print(self.weights)
                        
                
    def evaluate(self, testing_data, testing_labels):
        temp = faceArray(testing_data)
        ltemp = facelabelArray(testing_labels)
        count = 0
        for i in range(len(temp)):
            prediction = self.predict(temp[i])
            if prediction >= 0:
                result = 1
            else:
                result = 0
            if(result == int(ltemp[i])):
                count += 1
        percentcorrect = count/len(temp)
        print("Correct:", percentcorrect*100, "%")
        return percentcorrect
                
                    
 
        
        
   
def splitArray(inputs):
    #self  will refer to the stuff in init inputs will be the the given 28x28 array
    tempArr = np.array(inputs)
    #split the array into 10 6x70 arrays
    splitArr = np.hsplit(tempArr, 10)
    #split the array into 100 10x10 
    for i in range(10):
        splitArr[i] = np.array_split(splitArr[i], 10)
    #transforms each 4x4 into a 16 1D array so ending up with 49 length 16 vectors for the weights
    for i in range(10):
        for j in range(10):
            splitArr[i][j] = splitArr[i][j].flatten()
    return splitArr                   
            
    
        
    
#reads in a digit image file and returns a list of 28x28 matrices with " "= 0, # = 1, and + = 2 WE CAN CHANGE THE + TO 1 AS WELL IF LAZY (this will matter much more in the prediction function,  this may change the %success rate)
#this is new btw much better that I read the instructions a 15th time
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
            current_segment.append([1 if ch == "#" else 0 for ch in line.rstrip("\n")])
        if current_segment:
            segments.append(current_segment)        
    return segments


#reads in a file containing the labels of the corresponding images [1,2,...]
def facelabelArray(filename):
    with open(filename) as file:
        lines = file.readlines()
    label_arr = []
    for line in lines:
        label_arr.append(line.rstrip("\n"))
    return label_arr    

X = faceArray("facedatatrain.txt")
Y = splitArray(X[0])

p = np.zeros(100)
print(Y[4][0])
l = sum(np.dot(p[40],Y[4][0]))
print(l)
# epic = 0
        
# for i in range(10):
#     for j in range(10):
#         epic += (sum(np.dot(p[i],Y[i][j])))  
#     # epic.append(sum(dot)
# print(epic)
# Y = splitArray(X)
# for i in range(len(Y[0][0])):
    # print(Y[0][0][i])
digitP = SelfPerceptron()

digitP.train("facedatatrain.txt", "facedatatrainlabels.txt")
digitP.evaluate("facedatatest.txt", "facedatatestlabels.txt")
digitP.evaluate("facedatavalidation.txt", "facedatavalidationlabels.txt")

