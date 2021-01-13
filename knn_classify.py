# Name      :   Pranav Bhandari
# Student ID:   1001551132
# Date      :   11/04/2020

import sys, math, numpy as np

class Data:
    def __init__(self, distance, classlabel):
        self.distance = distance
        self.classlabel = classlabel

    def __lt__(self, other):
        return self.distance < other.distance
    
    def __str__(self):  
        return str(self.classlabel)


# This function is used to process the data before the algorithm.
# All the attributes and the class labels are normalized using their
# means and standard deviations.
def preprocess(filename):
    file = open(filename, "r")
    data = []
    means = []
    stdevs = []
    totalExamples = 0
    numAttributes = 0
    # flag variable is created to initialize the means and stdevs list only once.
    flag = 0
    for row in file:
        intermediate = []
        temp = row.split()
        # When the loop runs for the first time, the means and stdevs lists are initialized
        if flag == 0:
            numAttributes = len(temp)
            means = [0.0 for i in range(numAttributes-1)]
            stdevs = [0.0 for i in range(numAttributes-1)]
            flag = 1

        for i in range(numAttributes):
            val = float(temp[i])
            intermediate.append(val)
            if i!=numAttributes-1:
                means[i] += val
        data.append(intermediate)
    totalExamples = len(data)
    means = [float(num)/totalExamples for num in means]

    # Calculating the standard deviation for each attribute
    for row in data:
        for i in range(numAttributes-1):
            stdevs[i] += pow(row[i] - means[i], 2)

    stdevs = [pow((float(num)/(totalExamples-1)), 0.5) for num in stdevs]

    # Normalizing the data in terms of mean and stdev
    for i in range(totalExamples):
        for j in range(numAttributes-1):
            stdev = stdevs[j] if stdevs[j] !=0 else 1
            data[i][j] = float(data[i][j] - means[j])/stdev
    
    return data         

# Calculates the eucledian(L2) distance between two examples
def calculate_distance(data1, data2):
    dist = 0.0
    for i in range(len(data1)-1):
        dist += pow((data1[i] - data2[i]),2)
    dist = pow(dist, 0.5)
    return dist

def knn_classify(trainingfile, testfile, k):
    trainingdata = preprocess(trainingfile)
    testdata = preprocess(testfile)
    object_id = 1
    classification_accuracy = 0.0
    numAttributes = len(trainingdata[0])
    for test in testdata:
        neighbours = []
        for trainingExample in trainingdata:
            dist = calculate_distance(test, trainingExample)
            if len(neighbours) < k:
                neighbours.append(Data(dist, trainingExample[numAttributes-1]))
            else:
                neighbours.sort()
                if dist < neighbours[k-1].distance:
                    neighbours[k-1] = Data(dist, trainingExample[numAttributes-1])
        neighbours.sort()

        max_value = 0
        max_class = -1
        ties = 0
        ties_arr = []
        for i in range(k):
            value = neighbours[i].classlabel
            count = 0

            for j in range(k):
                if neighbours[j].classlabel == value:
                    count +=1
            
            if count > max_value:
                max_value = count
                max_class = value
                ties = 0
                ties_arr = []
                ties_arr.append(value)
            elif count == max_value and max_class!=value:
                ties +=1
                ties_arr.append(value)
        true_class = test[numAttributes-1]
        accuracy = 0.0 if ties==0 and true_class!= max_class else 1.0
        if ties!=0 and true_class in ties_arr:
            accuracy = 1/(ties+1)
        print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(object_id, int(max_class), int(true_class), accuracy))   
        object_id +=1
        classification_accuracy += accuracy
    print("classification accuracy={:6.4f}".format(classification_accuracy/(object_id-1)))

if __name__ == '__main__':
    knn_classify(sys.argv[1], sys.argv[2], int(sys.argv[3]))