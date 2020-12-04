To compile this file -> g++ decisionTreeSimple.cpp -fopenmp\
To run this program -> ./a.out -i Dataset1/Train.txt -m Dataset1/Test.txt

Run time options\
 -i trainingData   : training data file to build the decision tree (default TrainingSynthetic.txt)\
 -m testingData  : testing data file to test the accuracy of the decision tree (default TrainingSynthetic.txt)
 
trainingData text file stores the data points.\
    Each line contains the ID and coordinates of a single data point\
 The number of coordinates must be equal for all data points\
 Data is organized as list of attributes, last attribute is the class\
 each attribute is assumed to be CATEGORICAL (not numeric), however, represented as INT\
 class is assumed to be of type INT\
 Pre-processing of data may be required to match the input data format
 
testingData follows the same format as the trainingData

Assumption on decision trees\
 Decision Tree output is N dimensional -> split across attribute values\
 Number of leaf nodes is same as the number of unique attribute values\
 No assumption made for BINARY nature of the tree.\
 Number of children capped at 20 (can be changed for allocation of memory)
 
Debug flag -> if 1, prints intermediate data to validate progress\
 To set flag to 0, for performance metrics comparison purpose