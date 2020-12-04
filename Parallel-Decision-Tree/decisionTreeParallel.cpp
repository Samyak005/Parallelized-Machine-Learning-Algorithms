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
#include <mpi.h>

using namespace std;

/*
To compile this file -> mpic++ decisionTreeParallel.cpp -fopenmp
To run this program -> mpirun -host localhost -np 8 ./a.out -i Dataset1/Train.txt -m Dataset1/Test.txt 

Run time options
	-np nproc : Number of parallel processors
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

Assumption on decision tree
	Decision Tree output is N dimensional -> split across attribute values
	Number of leaf nodes is same as the number of unique attribute values
	Number of children capped at 20 (can be changed- done for allocation of memory)

Notes on MPI
	Number of processors as command line input should exceed the number of attributes
	For determining an attribute to split on, unique processor is assigned for each attribute
	And all the information gain values are collated to determine the attribute to split on.
	Observation : On a dual core computer time taken came out to be higher than a simple decision tree
	as for the synthetic dataset number of attributes is 6. Performance improved on 8-core machine.

Debug flag -> if 1, prints intermediate data to validate progress
    To set flag to 0, for performance metrics comparison purpose
*/

char *trainingData = "TrainingSynthetic.txt";
char *testingData = "TrainingSynthetic.txt";
int minLeafSize = 5;
int debug = 0;
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
	int level;			 // level of the node, distance from the root
	struct dtNode *child[20];
	int nChild = 0;			 // number of children, 0 for leaf node
	int attributeVal = -1;	 // value of the splitAttribute that defines the path
	int splitAttribute = -1; // spliting attribute, -1 for leaf node
};

// function to read training data and store in trainingData1
void readTrainingCSV()
{
	// input file stream (inputFileStrm) for reading data from file
	string readLine, token;

	ifstream inputFileStrm(trainingData, ios::in);
	int flag = 0;

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
			dp1.push_back(data[i]);
			dataElements.insert(make_pair(attributeVal, dp1));
			if (debug2)
				printf("Adding Attribute\n");
			attrUniqueValues.push_back(attributeVal);
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
int select(int level, vector<int> &attrUsed, vector<int> data, MPI_Comm comm)
{
	int i, selectedAttribute = -1;
	double iGainAttribute, maxVal;
	maxVal = threshold;

	int rank, nproc;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	// if (debug)
	// 	printf("Rank %d: in Select numAttributes %d \n", rank, numOfAttrib);
	// Each processor is asked to compute for a specific attribute
	if (rank == 0)
	{
		i = 1;
		while (i < numOfAttrib - 1)
		{
			// attribute has not been used earlier to split the data only then compute the options
			if (attrUsed[i] == 0)
				MPI_Send(&i, 1, MPI_INT, i, 0, comm);
			i++;
		}
	}

	i = 1;
	while (i < numOfAttrib - 1)
	{
		// Processor i will receive only the attribute i
		// and then compute information gain for that attribute
		// and then send back the information gain value to rank=0
		if ((rank == i) && (attrUsed[i] == 0))
		{
			MPI_Recv(&i, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
			iGainAttribute = getIGAttribute(i, data);
			if (debug)
			{
				printf("Gain: %f, Rank: %d,  Attr: %d\n", iGainAttribute, rank, i);
			}
			MPI_Send(&iGainAttribute, 1, MPI_DOUBLE, 0, 0, comm);
		}
		i++;
	}

	// Collate information gain value from each process in Process 0
	if (rank == 0)
	{
		i = 1;
		while (i < numOfAttrib - 1)
		{
			// attribute has not been used earlier to split the data only then compute the options
			if (attrUsed[i] == 0)
			{
				MPI_Recv(&iGainAttribute, 1, MPI_DOUBLE, i, 0, comm, MPI_STATUS_IGNORE);
				if (debug2)
					printf("iGainAttribute received from %d\n", i);
				if (iGainAttribute > maxVal)
				{
					maxVal = iGainAttribute;
					if (debug2)
						printf("maxValue changed, new maxValue is %f\n", maxVal);
					selectedAttribute = i;
				}
			}
			i++;
		}
		i = 1;
		while (i < numOfAttrib - 1)
		{
			MPI_Send(&selectedAttribute, 1, MPI_INT, i, 0, comm);
			MPI_Send(&maxVal, 1, MPI_DOUBLE, i, 0, comm);
			i++;
		}
	}
	else
	{
		MPI_Recv(&selectedAttribute, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
		if (debug2)
			printf("selectedAttribute message received by %d\n", rank);
		MPI_Recv(&maxVal, 1, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
		if (debug2)
			printf("maxValue message received by %d\n", rank);
	}
	// threshold defined to ensure splitting does not go beyond limit TO REVIEW LATER??
	if (maxVal <= threshold)
		return -1;

	// mark selectedAttribute as used
	attrUsed[selectedAttribute] = 1;

	if ((debug) && (rank == 0))
	{
		printf("level %d Rank %d Splitting Attribute:%d\n", level, rank, selectedAttribute);
	}
	return selectedAttribute;
}

// function for returning most probable output class
int determineClass(vector<int> data)
{
	if (data.size() == 1)
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
void buildTree(int level, vector<int> attrUsed, vector<int> data, dtNode *root, MPI_Comm comm)
{
	int selectedAttribute, i;

	int rank, nproc;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

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

	selectedAttribute = select(level, attrUsed, data, comm);
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
		if ((debug) && (rank == 0))
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
			buildTree(level + 1, attrUsed, dataElements[attributeVal], childrenNode, comm);
		}
		else
		{
			childrenNode->classLabel = determineClass(dataElements[attributeVal]);
			if ((debug) && (rank == 0))
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

	printf("output decision tree:\n");
	while (dq.size() != 0)
	{
		nnode = &(dq.front());
		dq.pop();
		for (int i = 0; i < nnode->nChild; i++)
		{
			dq.push(*(nnode->child[i]));
			if (nnode->child[i]->splitAttribute == -1)
				printf("%d: Class %d ", nnode->child[i]->level, nnode->child[i]->classLabel);
			else
				printf("%d: %d ", nnode->child[i]->level, nnode->child[i]->splitAttribute);
		}
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

	string readLine, token;
	ifstream inputFileStrm(testingData, ios::in);

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

	int rank, nproc, mpi_namelen;
	char mpi_name[MPI_MAX_PROCESSOR_NAME];
	double timing, io_timing, training_time;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Get_processor_name(mpi_name, &mpi_namelen);

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

	if (debug)
		printf("Proc %d of %d running on %s\n", rank, nproc, mpi_name);

	MPI_Barrier(MPI_COMM_WORLD);
	io_timing = MPI_Wtime();

	int i;

	readTrainingCSV();
	if ((rank == 0) && (debug))
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

	timing = MPI_Wtime();
	io_timing = timing - io_timing;
	training_time = timing;

	buildTree(0, attrUsed, data, root, MPI_COMM_WORLD);

	timing = MPI_Wtime();
	training_time = timing - training_time;

	if (is_output_timing)
	{
		printf("rank %d in output time computation %f\n", rank, training_time);
	}

	if ((rank == 0) && (debug2))
	{
		printf("output decision tree\n");
		outputDT(root);
	}
	// test decision tree
	test(root);

	//MPI_Finalize();
	return 0;
}