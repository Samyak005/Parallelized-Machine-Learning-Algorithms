For sequential:

reading data points from file DataSet/TestData100.txt
0: total distance of points from clusters 1203.314453
1: object 17 assigned new assignedClusterId 3 -> 1
1: object 26 assigned new assignedClusterId 3 -> 0
1: object 31 assigned new assignedClusterId 3 -> 1
1: object 32 assigned new assignedClusterId 3 -> 1
1: object 45 assigned new assignedClusterId 3 -> 1
1: object 46 assigned new assignedClusterId 3 -> 1
1: object 55 assigned new assignedClusterId 3 -> 0
1: object 57 assigned new assignedClusterId 0 -> 2
1: object 92 assigned new assignedClusterId 1 -> 3
1: object 96 assigned new assignedClusterId 3 -> 1
1: total distance of points from clusters 737.011963
2: object 8 assigned new assignedClusterId 3 -> 1
2: object 39 assigned new assignedClusterId 3 -> 1
2: object 72 assigned new assignedClusterId 0 -> 2
2: object 78 assigned new assignedClusterId 0 -> 2
2: object 93 assigned new assignedClusterId 3 -> 1
2: total distance of points from clusters 719.274231
3: object 15 assigned new assignedClusterId 0 -> 2
3: object 19 assigned new assignedClusterId 3 -> 1
3: object 20 assigned new assignedClusterId 3 -> 1
3: object 87 assigned new assignedClusterId 3 -> 1
3: object 90 assigned new assignedClusterId 3 -> 1
3: total distance of points from clusters 706.809814
4: object 9 assigned new assignedClusterId 3 -> 1
4: object 14 assigned new assignedClusterId 3 -> 1
4: object 50 assigned new assignedClusterId 3 -> 1
4: object 67 assigned new assignedClusterId 3 -> 1
4: object 86 assigned new assignedClusterId 3 -> 1
4: object 88 assigned new assignedClusterId 3 -> 1
4: total distance of points from clusters 696.148010
5: object 5 assigned new assignedClusterId 3 -> 1
5: object 59 assigned new assignedClusterId 3 -> 1
5: object 77 assigned new assignedClusterId 3 -> 1
5: total distance of points from clusters 684.840820
6: object 48 assigned new assignedClusterId 3 -> 1
6: object 85 assigned new assignedClusterId 3 -> 1
6: total distance of points from clusters 679.570984
7: object 98 assigned new assignedClusterId 3 -> 1
7: total distance of points from clusters 677.487976
8: total distance of points from clusters 676.120239
Computation time = 0.001769 sec
Writing coordinates of K=4 cluster centers to file "DataSet/TestData100.txt.seq_cluster_centres"
Writing assignedClusterId of N=100 data dataObjects to file "DataSet/TestData100.txt.seq_assignedClusterId"

For parallel:

reading data points from file DataSet/TestData100.txt
startIndex 0 endIndex 24
startIndex 25 endIndex 49
startIndex 50 endIndex 74
startIndex 75 endIndex 99
0: total distance of points from clusters 1203.314453
1: process id 1, object 26 assigned new assignedClusterId 3 -> 0
1: process id 2, object 55 assigned new assignedClusterId 3 -> 0
1: process id 1, object 31 assigned new assignedClusterId 3 -> 1
1: process id 1, object 32 assigned new assignedClusterId 3 -> 1
1: process id 2, object 57 assigned new assignedClusterId 0 -> 2
1: process id 0, object 17 assigned new assignedClusterId 3 -> 1
1: process id 1, object 45 assigned new assignedClusterId 3 -> 1
1: process id 1, object 46 assigned new assignedClusterId 3 -> 1
1: process id 3, object 92 assigned new assignedClusterId 1 -> 3
1: process id 3, object 96 assigned new assignedClusterId 3 -> 1
1: total distance of points from clusters 737.012085
2: process id 3, object 78 assigned new assignedClusterId 0 -> 2
2: process id 0, object 8 assigned new assignedClusterId 3 -> 1
2: process id 1, object 39 assigned new assignedClusterId 3 -> 1
2: process id 3, object 93 assigned new assignedClusterId 3 -> 1
2: process id 2, object 72 assigned new assignedClusterId 0 -> 2
2: total distance of points from clusters 719.274048
3: process id 0, object 15 assigned new assignedClusterId 0 -> 2
3: process id 3, object 87 assigned new assignedClusterId 3 -> 1
3: process id 3, object 90 assigned new assignedClusterId 3 -> 1
3: process id 0, object 19 assigned new assignedClusterId 3 -> 1
3: process id 0, object 20 assigned new assignedClusterId 3 -> 1
3: total distance of points from clusters 706.809692
4: process id 2, object 50 assigned new assignedClusterId 3 -> 1
4: process id 0, object 9 assigned new assignedClusterId 3 -> 1
4: process id 3, object 86 assigned new assignedClusterId 3 -> 1
4: process id 3, object 88 assigned new assignedClusterId 3 -> 1
4: process id 0, object 14 assigned new assignedClusterId 3 -> 1
4: process id 2, object 67 assigned new assignedClusterId 3 -> 1
4: total distance of points from clusters 696.148010
5: process id 0, object 5 assigned new assignedClusterId 3 -> 1
5: process id 3, object 77 assigned new assignedClusterId 3 -> 1
5: process id 2, object 59 assigned new assignedClusterId 3 -> 1
5: total distance of points from clusters 684.840698
6: process id 3, object 85 assigned new assignedClusterId 3 -> 1
6: process id 1, object 48 assigned new assignedClusterId 3 -> 1
6: total distance of points from clusters 679.571045
7: process id 3, object 98 assigned new assignedClusterId 3 -> 1
7: total distance of points from clusters 677.488098
8: total distance of points from clusters 676.120239
 0: numChangesProcess=0.000000 threshold=0.001000 loop=8
Computation time = 0.000858 sec
Computation time = 0.000855 sec
Computation time = 0.000853 sec
Computation time = 0.000853 sec
Writing coordinates of K=4 cluster centers to file "DataSet/TestData100.txt.cluster_centres"
Writing assignedClusterId of N=100 data dataObjects to file "DataSet/TestData100.txt.assignedClusterId"

