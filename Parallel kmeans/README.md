Run command:
 ```bash
 mpic++ mpikmeans.cpp -fopenmp
 ```
 ```bash
 mpirun -host localhost -np 4 ./a.out -n 4 -i DataSet/TestData100.txt 
 ```
For sequential
```bash
 mpirun -host localhost -np 1 ./a.out -n 4 -i DataSet/TestData100.txt
```
Run time options\
 -np nproc : Number of parallel processors\
 -i filename   : file containing data to be clustered\
 -n num_clusters: number of clusters (must be more than 1)\
 -t threshold   : threshold value (default 0.0010)
 
Input text file stores the data points to be clustered.
 1. Each line contains the ID and coordinates of a single data point
 2. The number of coordinates must be equal for all data points
 
Output files: There are two output files:
 1. Coordinates of cluster centers
    1. The file name is the input file name appended with ".cluster_centres".
    2. Each line contains an integer indicating the cluster id and the coordinates of the cluster center. 
 2. Membership of all data points to the clusters
    1. The file name is the input file name appended with ".assignedClusterId".
    2. each line contains two integers: data index (from 0 to the number of points)\
    and the cluster id indicating the assignedClusterId of the point.

Debug flag -> if 1, prints intermediate data to validate progress\
 To set flag to 0, for performance metrics comparison purpose