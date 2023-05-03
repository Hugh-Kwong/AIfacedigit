import numpy as np

numImgs = 0

class NaiveBayes:
    def __init__(self, input_size, learning_rate=0.1, num_epochs=100):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(49, 10)
        
def splitArray(inputs):
    #self  will refer to the stuff in init inputs will be the the given 60x70 array
    tempArr = np.array(inputs)
    #split the array into 7 4x2 arrays
    splitArr = np.hsplit(tempArr, 10)
    #split the array into 49 4x4 
    for i in range(10):
        splitArr[i] = np.array_split(splitArr[i], 10)
    #transforms each 4x4 into a 16 1D array so ending up with 49 length 16 vectors for the weights
    for i in range(10):
        for j in range(10):
            splitArr[i][j] = splitArr[i][j].flatten()
    return splitArr                   
        
def printArray(arr):
    for i in range(1):
        print()
        for j in range(70):
            print(arr[i][j])
        
def digit28Array(filename):
    segment_size = 70
    segments = []
    with open(filename, 'r') as f:
        current_segment = []
        for i, line in enumerate(f):
            segment_num = i // segment_size
            global numImgs
            numImgs = segment_num
            if i % segment_size == 0:
                if current_segment:
                    segments.append(current_segment)
                current_segment = []
            current_segment.append([1 if ch == "#" else 0 for ch in line.rstrip("\n")])
        if current_segment:
            segments.append(current_segment)        
    return segments
    
X = digit28Array("facedatatrain.txt")
printArray(X)
L = np.zeros((2, 100))
L = L+1
Y = np.array(X[0])
Y = np.hsplit(Y, 10)
# print(numImgs)
for i in range(10):
    Y[i] = np.array_split(Y[i], 10)
for i in range(10):
    for j in range(10):
        Y[i][j] = Y[i][j].flatten()
        
epic = []
for h in range(2):
    dot = []
    for i in range(10):
        for j in range(10):
            dot.append(sum(Y[i][j]))   
    epic.append(sum(dot))
temp2 = np.array(epic)
M = splitArray(X[0])


rows, cols = (100, 42)
isFace = [[0 for i in range(cols)] for j in range(rows)]
notFace = [[0 for i in range(cols)] for j in range(rows)]


# need to make a function that figures out whether the image is a face or not before trying to fill in the feature table to find the probability
print(sum(M[9][9]))
for i in range 10
    for j in range 10
        currFeature = i * 10 + j
        temp = sum(M[i][j])


for i in range(100):
    print(isFace[i])




# print(temp2)
# print(temp2[temp2.argmax()])

