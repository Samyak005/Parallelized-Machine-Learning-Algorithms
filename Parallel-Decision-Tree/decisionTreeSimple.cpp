#include <vector>
#include <map>
#include <queue>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <time.h>
#include <set>
#include <climits>
#include <omp.h>

using namespace std;

/*
To compile this file -> g++ decisionTreeSimple.cpp -fopenmp
To run this program -> ./a.out -i Dataset1/Train.txt -m Dataset1/Test.txt 

Run time options
    -i trainingData   : training data file to build the decision tree (default TrainingSynthetic.txt)
	-m testingData	 : testing data file to test the accuracy of the decision tree (default TrainingSynthetic.txt)

trainingData text file stores the data points. 
    Each line contains the ID and coordinates of a single data point
    The number of coordinates must be equal for all data points
    Data is organized as list of attributes, last attribute is the class
	each attribute is assumed to be CATEGORICAL (not numeric), however, represented as INT
	class is assumed to be of type INT
	Pre-processing of data may be required to match the input data format

testingData follows the same format as the trainingData

Assumption on decision trees
	Decision Tree output is N dimensional -> split across attribute values
	Number of leaf nodes is same as the number of unique attribute values
	No assumption made for BINARY nature of the tree.
	Number of children capped at 20 (can be changed for allocation of memory)

Debug flag -> if 1, prints intermediate data to validate progress
    To set flag to 0, for performance metrics comparison purpose
*/

char *trainingData = "TrainingSynthetic.txt";
char *testingData = "TrainingSynthetic.txt";
int minLeafSize = 5;
int debug = 1;
int debug2 = 0;
float threshold = 0.01;

// 2d vector to store training data
vector<vector<int>> trainingData1;
vector<int> classValues;

// vector to store row number for data in file
//(data does not transferred, only row numbers are transferred in function calls)
vector<int> data;
// vector to store if attribute has already been used or not
vector<int> attrUsed;

int numOfAttrib = 0;
int numOfDataEle = 0;
long int sim_init_time = 0;

long int gettimeelapsed()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (ts.tv_nsec - sim_init_time);
}

int setstarttime()
{
    struct timespec ts_sim;
    clock_gettime(CLOCK_REALTIME, &ts_sim);
    sim_init_time = ts_sim.tv_nsec;
    return sim_init_time;
}

struct dtNode
{
    int classLabel = -1; // class value at leaf node, -1 for decision node
    int level;           // level of the node, distance from the root
    struct dtNode *child[20];
    int nChild = 0;          // number of children, 0 for leaf node
    int attributeVal = -1;   // value of the splitAttribute that defines the path
    int splitAttribute = -1; // spliting attribute, -1 for leaf node
};

// function to read training data and store in trainingData1
void readTrainingCSV()
{
    // input file stream (inputFileStrm) for reading data from file
    string readLine, token;

    ifstream inputFileStrm(trainingData, ios::in);
    int flag = 0;
    if (!inputFileStrm.is_open())
    {
        printf("file open failed");
        return;
    }
    // read from inputFileStrm into string 'readLine'
    while (getline(inputFileStrm, readLine))
    {
        vector<int> rowData;
        stringstream listTokens(readLine);

        while (getline(listTokens, token, ','))
        {
            const char *s1 = token.c_str();
            rowData.push_back(atoi(s1));
            if (flag == 0)
            {
                attrUsed.push_back(0);
                numOfAttrib++; // compute it only once
            }
        }
        trainingData1.push_back(rowData);
        // bmpic++ decisionTreeParallel.cpp -fopenmpuild a list of unique class values
        int foundClass = 0;
        for (int i = 0; i < classValues.size(); i++)
        {
            if (classValues[i] == trainingData1[numOfDataEle][numOfAttrib - 1])
                foundClass = 1;
        }
        if (!foundClass)
            classValues.push_back(trainingData1[numOfDataEle][numOfAttrib - 1]);

        data.push_back(numOfDataEle);
        numOfDataEle++;
        flag = 1;
    }
    inputFileStrm.close();
}

double getIG(vector<int> data)
{
    double totalCount = data.size();
    double entropyValue = 0;

    // printf("data size %ld\n", data.size());
    // classStats: keeps count of each output class in data vector
    vector<double> classStats;
    for (int i = 0; i < classValues.size(); i++)
    {
        classStats.push_back(0);
    }

    for (int i = 0; i < totalCount; i++)
    {
        // find this class value if encountered earlier or not
        for (int j = 0; j < classValues.size(); j++)
        {
            if (classValues[j] == trainingData1[data[i]][numOfAttrib - 1])
                classStats[j]++;
        }
    }

    for (int i = 0; i < classValues.size(); i++)
    {
        if (classStats[i] != 0)
            entropyValue += (classStats[i] / totalCount) * (log(classStats[i] / totalCount) / log(2));
    }
    entropyValue = -1 * entropyValue;
    return entropyValue;
}

// Calculate information gain for an attribute
double getIGAttribute(int attr, vector<int> data)
{
    int i, attributeVal;
    int dataSize = data.size();
    double attrInfoGain = 0;

    map<int, vector<int>> dataElements;
    vector<int> attrUniqueValues;

    for (int i = 0; i < data.size(); i++)
    {
        vector<int> dp1;

        int attributeVal = trainingData1[data[i]][attr];
        int foundAttribute = 0;
        for (int i = 0; i < attrUniqueValues.size(); i++)
        {
            if (attrUniqueValues[i] == attributeVal)
                foundAttribute = 1;
        }
        if (!foundAttribute)
        {
            attrUniqueValues.push_back(attributeVal);
            dp1.push_back(data[i]);
            dataElements.insert(make_pair(attributeVal, dp1));
            if (debug2)
                printf("Adding Attribute\n");
        }
        else
        {
            dataElements[attributeVal].push_back(data[i]);
        }
    }

    for (int i = 0; i < attrUniqueValues.size(); i++)
    {
        // compute the weighted average sum of information gain for different attribute values of the attribute
        attributeVal = attrUniqueValues[i];
        if (debug)
            printf("Attribute %d Attribute Value %d Data Count %f\n", attr, attributeVal, (double)dataElements[attributeVal].size());
        attrInfoGain += ((double)dataElements[attributeVal].size() / (double)dataSize) * getIG(dataElements[attributeVal]);
    }
    return getIG(data) - attrInfoGain;
}

// function to determine the splitting attribute
// attr: candidate attributes for splitting attribute, attrUsed[i]=1 if already used
// data: data row nos(in the file and index in "trainingData1" vector) used for calculating information gains
//
// Each process separately and independently evaluates one attribute and shares the outcome
// The training data is available to all processes,
// communication between two steps is the attribute to evaluate
// Each process returns the Information Gain corresponding to that attribute
// Hence inter process communication is kept at the minimum
int select(int level, vector<int> &attrUsed, vector<int> data)
{
    int i, selectedAttribute = -1;
    double iGainAttribute, maxVal;
    maxVal = threshold;

    i = 1;
    while (i < numOfAttrib - 1)
    {
        if (attrUsed[i] == 0)
            iGainAttribute = getIGAttribute(i, data);
        if (debug)
        {
            printf("Gain: %f, Attr: %d\n", iGainAttribute, i);
        }
        if (iGainAttribute > maxVal)
        {
            maxVal = iGainAttribute;
            if (debug2)
                printf("maxValue changed, new maxValue is %f\n", maxVal);
            selectedAttribute = i;
        }
        i++;
    }

    if (maxVal <= threshold)
        return -1;

    // mark selectedAttribute as used
    attrUsed[selectedAttribute] = 1;

    if (debug)
    {
        printf("level %d Splitting Attribute:%d\n", level, selectedAttribute);
    }
    return selectedAttribute;
}

// function for returning most probable output class
int determineClass(vector<int> data)
{
    if (data.size() ==1 )
    {
        return trainingData1[data[0]][numOfAttrib - 1];
    }
    int maxClass = -1, maxVal;
    // classStats: keeps count of each output class in data vector
    vector<int> classStats;
    for (int i = 0; i < classValues.size(); i++)
    {
        classStats.push_back(0);
    }

    for (int i = 0; i < data.size(); i++)
    {
        // find this class value if encountered earlier or not
        for (int i = 0; i < classValues.size(); i++)
        {
            if (classValues[i] == trainingData1[data[i]][numOfAttrib - 1])
                classStats[i]++;
        }
    }
    maxVal = threshold;
    // ans contains determineClass
    for (int i = 0; i < classValues.size(); i++)
    {
        //if (debug) printf("%d classStats %d\n",i,classStats[i]);
        if (classStats[i] > maxVal)
        {
            maxVal = classStats[i];
            maxClass = classValues[i];
        }
    }
    return maxClass;
}

// Recursive splitting of the data
void buildTree(int level, vector<int> attrUsed, vector<int> data, dtNode *root)
{
    int selectedAttribute, i;

    // if all class values are same
    int nClasses = 1;
    i = 1;
    int oneClass = trainingData1[data[0]][numOfAttrib - 1];
    while ((i < data.size()) && (nClasses <= 1))
    {
        if (trainingData1[data[i]][numOfAttrib - 1] != oneClass)
            nClasses++;
        i++;
    }
    if (nClasses == 1)
    {
        root->classLabel = oneClass;
        return;
    }

    selectedAttribute = select(level, attrUsed, data);
    root->splitAttribute = selectedAttribute;

    if (selectedAttribute == -1)
    {
        root->classLabel = determineClass(data);
        if (debug2)
            printf("new classlabel assigned %d\n", root->classLabel);
        return;
    }

    map<int, vector<int>> dataElements;
    vector<int> attrUniqueValues;
    for (int i = 0; i < data.size(); i++)
    {
        int attributeVal = trainingData1[data[i]][selectedAttribute];
        int foundAttribute = 0;
        for (int i = 0; i < attrUniqueValues.size(); i++)
        {
            if (attrUniqueValues[i] == attributeVal)
                foundAttribute = 1;
        }
        if (!foundAttribute)
        {
            vector<int> dp1;
            dp1.push_back(data[i]);
            dataElements.insert(make_pair(attributeVal, dp1));
            attrUniqueValues.push_back(attributeVal);
        }
        else
        {
            dataElements[attributeVal].push_back(data[i]);
        }
    }

    root->nChild = attrUniqueValues.size();
    for (int i = 0; i < attrUniqueValues.size(); i++)
    {
        int attributeVal = attrUniqueValues[i];
        if (debug)
        {
            printf("Recursive call -> level %d: attribute %d attributeVal %d #data Points %ld:\n", level + 1, selectedAttribute, attributeVal, dataElements[attributeVal].size());
        }

        dtNode *childrenNode = new dtNode;
        childrenNode->attributeVal = attributeVal;
        childrenNode->splitAttribute = -1;
        childrenNode->nChild = 0;
        childrenNode->classLabel = -1;

        root->child[i] = childrenNode;
        childrenNode->level = level + 1;

        if (dataElements[attributeVal].size() > minLeafSize)
        {
            buildTree(level + 1, attrUsed, dataElements[attributeVal], childrenNode);
        }
        else
        {
            childrenNode->classLabel = determineClass(dataElements[attributeVal]);
            if (debug)
            {
                printf("Leaf -> level %d, #data Points %ld Class Label %d:\n", level + 1, dataElements[attributeVal].size(), childrenNode->classLabel);
            }
            childrenNode->nChild = 0;
        }
    }
    return;
}

void outputDT(dtNode *root)
{
    dtNode *nnode;
    queue<dtNode> dq;
    printf("%d:%d\n", root->level, root->splitAttribute);
    dq.push(*root);

    //printf("output decision tree:\n");
    while (dq.size() != 0)
    {
        nnode = &(dq.front());
        dq.pop();
        if (debug)
            printf("level %d #child %d\n", nnode->level, nnode->nChild);
        int i = 0;
        while (i < nnode->nChild)
        {
            dq.push(*(nnode->child[i]));
            if (nnode->child[i]->splitAttribute == -1)
                printf("%d:Class %d ", nnode->child[i]->level, nnode->child[i]->classLabel);
            else
                printf("%d: %d ", nnode->child[i]->level, nnode->child[i]->splitAttribute);

            i++;
        }
        if (debug)
            printf(" Queue size %ld\n", dq.size());
        else
            printf("\n");
    }
    return;
}

void test(dtNode *root)
{
    int i, j;
    int attr, attrVal, flag;
    dtNode *dnode;
    int numCorrect = 0;
    int numIncorrect = 0;
    int numUnexpected = 0;
    vector<int> rowData;

    if (debug)
        printf("in testing\n");

    string readLine, token;
    ifstream inputFileStrm(testingData, ios::in);
    if (!inputFileStrm.is_open())
    {
        printf("file open failed");
        return;
    }

    while (getline(inputFileStrm, readLine))
    {
        rowData.clear();
        stringstream listTokens(readLine);
        while (getline(listTokens, token, ','))
        {
            const char *s1 = token.c_str();
            rowData.push_back(atoi(s1));
        }

        dnode = root;
        //traverse decision tree
        while (dnode->classLabel == -1 && dnode->splitAttribute != -1)
        {
            for (j = 0; j < dnode->nChild; j++)
            {
                if (dnode->child[j]->attributeVal == rowData[dnode->splitAttribute])
                    break;
            }
            if (j == dnode->nChild)
            {
                if (dnode->classLabel == -1)
                    numUnexpected++;
                break;
            }
            else
                dnode = dnode->child[j];
        }
        if (dnode->classLabel == rowData[numOfAttrib - 1])
            numCorrect++;
        else
            numIncorrect++;
    }
    inputFileStrm.close();
    printf("Correct %d Incorrect %d Unexpected %d\n", numCorrect, numIncorrect, numUnexpected);
    return;
}

int main(int argc, char **argv)
{
    int opt;
    extern char *optarg;
    int is_output_timing = 1;

    while ((opt = getopt(argc, argv, "i:m:n")) != EOF)
    {
        switch (opt)
        {
        case 'i':
            trainingData = optarg;
            break;
        case 'm':
            testingData = optarg;
            break;
        case 'n':
            minLeafSize = atoi(optarg);
            break;
        default:
            break;
        }
    }

    int i;

    readTrainingCSV();
    if (debug)
    {
        printf("Number of Data Elements %d\n", numOfDataEle);
        printf("Number of attributes %d\n", numOfAttrib);
        printf("Number of class values %ld\n", classValues.size());
    }

    dtNode *root = new dtNode;
    root->nChild = 0;
    root->splitAttribute = -1;
    root->classLabel = -1;
    root->level = 0;
    root->attributeVal = -1;

    double start = omp_get_wtime();
    buildTree(0, attrUsed, data, root);
    double end = omp_get_wtime();
    if (is_output_timing)
        printf("time:%f\n", end - start);

    if (debug2) 
    {
        printf("output decision tree\n");
        outputDT(root);
    }
    // test decision tree
    test(root);

    return 0;
}