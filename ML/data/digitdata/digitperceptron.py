import numpy as np


class SelfPerceptron:
    #weights will change to an array of 49 weights corresponding to each 4x4 split array look at comments on predict
    def __init__(self, input_size = 27, learning_rate=0.1, num_epochs=2):
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
            count = 0
            for i in range(7):
                for j in range(7):
                    dot.append(sum(np.dot(self.weights[h][count],splitArr[i][j])))  
                    count += 1  
            epic.append(sum(dot))
        temp2 = np.array(epic)
        return(temp2.argmax())
             
            

    #to train we run through the array of images for epoch amount of iterations and call prediction on each iteration. The weights will be changed depend on the error given by the prediction.
    #we can count the number of arrays with len(array)
    #self is self, training data is the name of the data file, training labels is file of corresponding labels.
    def train(self, training_data, training_labels, percent):
        temp = digit28Array(training_data)
        ltemp = digitlabelArray(training_labels)
        #i is each array, 
        for z in range(2):
            for i in range(int(len(temp) * percent/10)):
                prediction = self.predict(temp[i])
                # print(prediction, ltemp[i])
                split = splitArray(temp[i])
                # print(split[0][0])
                if prediction !=  ltemp[i]:
                    count = 0
                    label = ltemp[i]
                    # print("EPIC STYLE", prediction, label)
                    for j in range(7):
                        for k in range(7):
                            self.weights[prediction][count] -= sum(split[j][k])
                            self.weights[int(label)][count] += sum(split[j][k])
                            count += 1
        # print(self.weights)
                        
                
    def evaluate(self, testing_data, testing_labels):
        temp = digit28Array(testing_data)
        ltemp = digitlabelArray(testing_labels)
        count = 0
        for i in range(len(temp)):
            prediction = self.predict(temp[i])
            if(prediction == int(ltemp[i])):
                count += 1
        percentcorrect = count/len(temp)
        print("Correct:", percentcorrect*100, "%")
        return percentcorrect
                
                    
 
        
            
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



for i in range(10):
    print((i+1)*10,"%")
    digitP = SelfPerceptron()
    digitP.train("trainingimages.txt", "traininglabels.txt", i+1)
    digitP.evaluate("testimages.txt", "testlabels.txt")
