#include <stdio.h>
#include <vector>
#include <map>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>
#include <errno.h>

using namespace std;

/* Run command:
    mpic++ mpikmeans.cpp -fopenmp
    mpirun -host localhost -np 4 ./a.out -n 4 -i DataSet/TestData100.txt 

For sequential 
    mpirun -host localhost -np 1 ./a.out -n 4 -i DataSet/TestData100.txt 

Run time options
	-np nproc : Number of parallel processors
    -i filename   : file containing data to be clustered
    -n num_clusters: number of clusters (must be more than 1)
    -t threshold   : threshold value (default 0.0010)

Input text file stores the data points to be clustered. 
    o Each line contains the ID and coordinates of a single data point
    o The number of coordinates must be equal for all data points

Output files: There are two output files:
  * Coordinates of cluster centers
    o The file name is the input file name appended with ".cluster_centres".
    o each line contains an integer indicating the cluster id and the coordinates of the cluster center.
  * Membership of all data points to the clusters
    o The file name is the input file name appended with ".assignedClusterId".
    o each line contains two integers: data index (from 0 to the number of points) 
    and the cluster id indicating the assignedClusterId of the point.

Debug flag -> if 1, prints intermediate data to validate progress
    To set flag to 0, for performance metrics comparison purpose

*/

int debug = 1;
int debug2 = 0;
long int sim_init_time = 0;
char *filename = NULL;

#define MAX_ITERATIONS 500

int is_output_timing = 1;

double timing, io_timing, clustering_timing;

vector<vector<float>> dataObjects;
int numDimensions = 0;
int kk = 4; // number of cluster in K Means Clustering
int numObjs = 0;

// function to read data and store in dataObjects
void readCSV(char *filename)
{
    // input file stream (inputFileStrm) for reading data from file
    string readLine, token;

    ifstream inputFileStrm(filename, ios::in);

    if (!inputFileStrm.is_open())
    {
        printf("file open failed");
        return;
    }
    int flag = 0;
    numDimensions = 0;
    numObjs = 0;
    // read from inputFileStrm into string 'readLine'
    while (getline(inputFileStrm, readLine))
    {
        vector<float> rowData;
        stringstream listTokens(readLine);

        while (getline(listTokens, token, ','))
        {
            const char *s1 = token.c_str();
            rowData.push_back(atof(s1));
            if (flag == 0)
            {
                numDimensions++; // compute it only once
            }
        }
        dataObjects.push_back(rowData);
        numObjs++;
        flag = 1;
    }
    // printf("num dimensions %d\n", numDimensions);
    // printf("num Objects %d\n", numObjs);
    inputFileStrm.close();
    return;
}

__inline static float distance(vector<float> c1, float *c2)
{
    float euclid;
    euclid = 0.0;
    for (int j = 1; j < numDimensions; j++)
    {
        euclid = euclid + (c1[j] - c2[j]) * (c1[j] - c2[j]);
    }
    return euclid;
}

int mpi_kmeans(int numObjs, float threshold, MPI_Comm comm)
{
    int i, j, nproc, rank, index, loop = 0;

    int numObjectsProcessCluster[kk];
    int numObjectsCluster[kk];
    int startIndex[kk];
    int endIndex[kk];

    float numChangesProcess; // % of dataObjects change their clusters
    float numChangesAll;
    float newProcessClusters[kk][numDimensions];
    float newClusters[kk][numDimensions];
    float clusters[kk][numDimensions];

    timing = MPI_Wtime();
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    int divd = numObjs / nproc;
    int rem = numObjs % nproc;

    int assignedClusterId[numObjs];
    // assignedClusterId is the cluster id for each data object
    for (i = 0; i < numObjs; i++)
        assignedClusterId[i] = -1;

    for (i = 0; i < nproc; i++)
    {
        startIndex[i] = i * divd;
        endIndex[i] = (i + 1) * divd - 1;
        if ((debug) && (rank == 0))
            printf("startIndex %d endIndex %d\n", startIndex[i], endIndex[i]);
    }
    endIndex[nproc - 1] = numObjs - 1;

    printf("startIndex %d endIndex %d\n", startIndex[nproc - 1], endIndex[nproc - 1]);

    if (rank == 0)
    {
        if (debug2)
            printf("selecting the first %d elements as initial centers\n", kk);
        // copy the first kk elements in feature[]
        for (i = 0; i < kk; i++)
        {
            // srand(time(NULL));
            // int indx = rand() % numObjs;
            if (debug2)
                printf("Cluster %d initialized as the data points %d\n", i, i);
            for (j = 0; j < numDimensions; j++)
                clusters[i][j] = dataObjects[i][j];
        }
    }
    if (debug2)
        printf("Cluster Initialization done \n");
    MPI_Bcast(clusters[0], kk * numDimensions, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // initialize  clustersize
    for (i = 0; i < kk; i++)
    {
        numObjectsCluster[i] = 0;
        numObjectsProcessCluster[i] = 0;
        for (j = 0; j < numDimensions; j++)
        {
            newProcessClusters[i][j] = 0.0;
            newClusters[i][j] = 0.0;
        }
    }

    if (debug2)
        printf("%d: numObjs=%d kk=%d\n", rank, numObjs, kk);
    float numChanges;
    do
    {
        numChangesAll = 0.0;
        numChangesProcess = 0.0;
        numChanges = 0.0;
        float totaldistance = 0;
        for (i = startIndex[rank]; i <= endIndex[rank]; i++)
        {
            int index;
            float dist, min_dist;

            // find the cluster id that has min distance to object
            index = 0;
            min_dist = distance(dataObjects[i], clusters[0]);

            for (j = 1; j < kk; j++)
            {
                dist = distance(dataObjects[i], clusters[j]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    index = j;
                }
            }
            totaldistance += min_dist;

            // if assignedClusterId changes, increase numChangesProcess by 1
            if (assignedClusterId[i] != index)
            {
                numChangesProcess += 1.0;
                if ((debug) && (loop > 0))
                {
                    printf("%d: process id %d, object %d assigned new assignedClusterId %d -> %d\n", loop, rank, i, assignedClusterId[i], index);
                }
            }

            assignedClusterId[i] = index;
            numObjectsProcessCluster[index]++;

            if (debug2)
                printf("averaging for cluster center computation \n");
            for (j = 0; j < numDimensions; j++)
                newProcessClusters[index][j] = newProcessClusters[index][j] + dataObjects[i][j];
        }

        float aggDistance=0.0;
        MPI_Allreduce(&totaldistance, &aggDistance, 1, MPI_FLOAT, MPI_SUM, comm);

        if ((debug) && (rank==0))
            printf("%d: total distance of points from clusters %f\n", loop, aggDistance);

        // if ((rank == 0) && (debug2))
        //     for (i = 0; i < kk; i++)
        //     {
        //         printf("rank %d\n", rank);
        //         for (j = 0; j < numDimensions; j++)
        //             printf("%5f ", newProcessClusters[i][j]);
        //         printf("\n");
        //     }

        // sum all data dataObjects in newProcessClusters
        MPI_Allreduce(newProcessClusters[0], newClusters[0], kk * numDimensions, MPI_FLOAT, MPI_SUM, comm);
        MPI_Allreduce(numObjectsProcessCluster, numObjectsCluster, kk, MPI_INT, MPI_SUM, comm);

        // if ((rank == 0) && (debug2))
        //     for (i = 0; i < kk; i++)
        //     {
        //         printf("rank %d\n", rank);
        //         for (j = 0; j < numDimensions; j++)
        //             printf("%5f ", clusters[i][j]);
        //         printf("\n");
        //     }

        if ((debug2) && (rank == 0))
        {
            for (i = 0; i < kk; i++)
            {
                printf("Rank %d: Cluster size %d:%d", rank, i, numObjectsCluster[i]);
            }
            printf("\n");
        }

        // average the sum
        for (i = 0; i < kk; i++)
        {
            for (j = 0; j < numDimensions; j++)
            {
                clusters[i][j] = newClusters[i][j] / numObjectsCluster[i];
                newClusters[i][j] = 0;
                newProcessClusters[i][j] = 0.0;
            }
            numObjectsProcessCluster[i] = 0;
            numObjectsCluster[i] = 0;
        }

        if (debug2)
            printf("Clusters averaging done \n");
        MPI_Allreduce(&numChangesProcess, &numChangesAll, 1, MPI_FLOAT, MPI_SUM, comm);

        numChanges = numChangesAll / numObjs;

    } while (numChanges > threshold && loop++ < MAX_ITERATIONS);
    int newClusterId[numObjs];
    for (i = 0; i < numObjs; i++)
    {
        newClusterId[i] = -1;
    }

    MPI_Allreduce(assignedClusterId, newClusterId, numObjs, MPI_INT, MPI_MAX, comm);

    if ((debug) && rank == 0)
        printf("%2d: numChangesProcess=%f threshold=%f loop=%d\n", rank, numChangesProcess, threshold, loop);

    clustering_timing = MPI_Wtime();
    clustering_timing = clustering_timing - timing;

    if (is_output_timing)
        printf("Computation time = %f sec\n", clustering_timing);

    FILE *fptr;
    char outFileName[1024];

    // output: the coordinates of the cluster centres
    if (rank == 0)
    {
        sprintf(outFileName, "%s.cluster_centres", filename);
        printf("Writing coordinates of K=%d cluster centers to file \"%s\"\n", kk, outFileName);
        fptr = fopen(outFileName, "w");
        for (i = 0; i < kk; i++)
        {
            fprintf(fptr, "%d ", i);
            for (j = 1; j < numDimensions; j++)
                fprintf(fptr, "%f ", clusters[i][j]);
            fprintf(fptr, "\n");
        }
        fclose(fptr);

        // output: the closest cluster centre to each of the data points
        sprintf(outFileName, "%s.assignedClusterId", filename);
        printf("Writing assignedClusterId of N=%d data dataObjects to file \"%s\"\n", numObjs, outFileName);
        fptr = fopen(outFileName, "w");
        for (i = 0; i < numObjs; i++)
            fprintf(fptr, "%d %d\n", i, newClusterId[i]);
        fclose(fptr);
    }

    return 1;
}

int mpi_kmeans_sequential(int numObjs, float threshold)
{
    int i, j, index, loop = 0;

    int numObjectsCluster[kk];
    float numChangesAll;
    float clusters[kk][numDimensions];
    float newClusters[kk][numDimensions];

    timing = MPI_Wtime();

    int assignedClusterId[numObjs];
    // assignedClusterId is the cluster id for each data object
    for (i = 0; i < numObjs; i++)
        assignedClusterId[i] = -1;

    if (debug2)
        printf("selecting the first %d elements as initial centers\n", kk);
    // copy the first kk elements in feature[]
    for (i = 0; i < kk; i++)
    {
        // srand(time(NULL));
        // int indx = rand() % numObjs;
        if (debug2)
            printf("Cluster %d initialized as the data points %d\n", i, i);
        for (j = 0; j < numDimensions; j++)
        {
            clusters[i][j] = dataObjects[i][j];
            newClusters[i][j] = 0.0;
        }
    }

    if (debug2)
        printf("Cluster Initialization done \n");

    // initialize  clustersize
    for (i = 0; i < kk; i++)
        numObjectsCluster[i] = 0;

    if (debug2)
        printf("numObjs=%d kk=%d\n", numObjs, kk);

    float numChanges;
    do
    {
        numChangesAll = 0.0;
        numChanges = 0.0;
        float totaldistance = 0;
        for (i = 0; i < numObjs; i++)
        {
            int index;
            float dist, min_dist;

            // find the cluster id that has min distance to object
            index = 0;
            min_dist = distance(dataObjects[i], clusters[0]);

            for (j = 1; j < kk; j++)
            {
                dist = distance(dataObjects[i], clusters[j]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    index = j;
                }
            }
            totaldistance += min_dist;

            // if assignedClusterId changes, increase numChangesProcess by 1
            if (assignedClusterId[i] != index)
            {
                numChangesAll += 1.0;
                if ((debug) && (loop > 0))
                {
                    printf("%d: object %d assigned new assignedClusterId %d -> %d\n", loop, i, assignedClusterId[i], index);
                }
            }

            assignedClusterId[i] = index;
            numObjectsCluster[index]++;

            if (debug2)
                printf("averaging for cluster center computation \n");
            for (j = 0; j < numDimensions; j++)
                newClusters[index][j] = newClusters[index][j] + dataObjects[i][j];
        }

        if (debug)
            printf("%d: total distance of points from clusters %f\n", loop, totaldistance);

        // average the sum
        for (i = 0; i < kk; i++)
            for (j = 0; j < numDimensions; j++)
                clusters[i][j] = newClusters[i][j] / numObjectsCluster[i];

        // replace old cluster centers with newProcessClusters
        for (i = 0; i < kk; i++)
        {
            for (j = 0; j < numDimensions; j++)
                newClusters[i][j] = 0.0;
            numObjectsCluster[i] = 0;
        }

        numChanges = numChangesAll / numObjs;

    } while (numChanges > threshold && loop++ < MAX_ITERATIONS);

    clustering_timing = MPI_Wtime();
    clustering_timing = clustering_timing - timing;

    if (is_output_timing)
        printf("Computation time = %f sec\n", clustering_timing);

    FILE *fptr;
    char outFileName[1024];

    // output: the coordinates of the cluster centres

    sprintf(outFileName, "%s.seq_cluster_centres", filename);
    printf("Writing coordinates of K=%d cluster centers to file \"%s\"\n", kk, outFileName);
    fptr = fopen(outFileName, "w");
    for (i = 0; i < kk; i++)
    {
        fprintf(fptr, "%d ", i);
        for (j = 1; j < numDimensions; j++)
            fprintf(fptr, "%f ", clusters[i][j]);
        fprintf(fptr, "\n");
    }
    fclose(fptr);

    // output: the closest cluster centre to each of the data points
    sprintf(outFileName, "%s.seq_assignedClusterId", filename);
    printf("Writing assignedClusterId of N=%d data dataObjects to file \"%s\"\n", numObjs, outFileName);
    fptr = fopen(outFileName, "w");
    for (i = 0; i < numObjs; i++)
        fprintf(fptr, "%d %d\n", i, assignedClusterId[i]);
    fclose(fptr);

    return 1;
}

int main(int argc, char **argv)
{
    int opt;
    extern char *optarg;
    extern int optind;
    int i, j;
    int rank;
    int nproc;
    float threshold = 0.001;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    kk = 4;

    while ((opt = getopt(argc, argv, "i:n:t")) != EOF)
    {
        switch (opt)
        {
        case 'n':
            kk = atoi(optarg);
            break;
        case 'i':
            filename = optarg;
            break;
        case 't':
            threshold = atof(optarg);
            break;
        default:
            break;
        }
    }

    if (filename == 0 || kk <= 1)
    {
        printf("nothing to do.\n");
        MPI_Finalize();
        exit(1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // read data points from file
    if (rank == 0)
        printf("reading data points from file %s\n", filename);

    readCSV(filename);

    if ((debug2) && (rank == 0))
    {
        printf("File %s numObjs   = %d\n", filename, numObjs);
        printf("File %s numDimensions = %d\n", filename, numDimensions);
    }

    if (numObjs < kk)
    {
        if (rank == 0)
            printf("Error: number of clusters must be larger than the number of data points to be clustered.\n");
        MPI_Finalize();
        return 1;
    }
    // if (debug2)
    // {
    //     for (i = 0; i < kk; i++)
    //     {
    //         printf("dataObjects[%d]= ", i);
    //         for (j = 0; j < numDimensions; j++)
    //         {
    //             printf("%5f", dataObjects[i][j]);
    //         }
    //         printf("\n");
    //     }
    // }

    if (nproc > 1)
    {
        mpi_kmeans(numObjs, threshold, MPI_COMM_WORLD);
    }
    else
    {
        mpi_kmeans_sequential(numObjs, threshold);
    }

    MPI_Finalize();
    return (0);
}
