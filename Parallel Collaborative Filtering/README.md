To compile this file -> mpic++ ParallelMovieRecommendation.cpp -fopenmp\
To run this program -> mpirun -host localhost -np 8 ./a.out -n 5 -p 3245

Run time options\
   -np nproc : Number of parallel processors\
    -i movieDataFile : training data file to build the Movie Similarity Matrix (default movies.dat)\
    -r ratingsDataFile: training data file to build the Movie Similarity Matrix (default ratings.dat)\
    -n top_num : number of recommendations that should be shown for a user (default 5)\
    -p user   : user id for which the recommendations of movies should be shown (default value 3245)

Data of this program is the MovieLens 1M dataset from http://grouplens.org/datasets/movielens/\
    movies.dat stores the Movie Id, Movie Name, Tags, for example, 9::Sudden Death (1995)::Action\
    ratings.dat stores the Userid, Movie Id, Rating, Time, for example, 3123::2688::3::969323556

Collaborative Filtering\
    Item-based Collaborative Filtering implementation -> computes the Movie Similarity Matrix\
    In the training mode, Movie Similarity Matrix (nMovies * nMovies) is computed and stored as a file\
    Cosine is used to compute the similarity score\
    O (number of users) = ratings by all users for movie m1 * ratings by all users for movie m2.\
    In the testing mode, the matrix is loaded to recommend movie for users.

Notes on MPI\
   More the number of processors, more is the performance speed up\
    All processors are used equally to compute the Movie Similarityy Matrix (compute allocated using % mod)\
    And MPI_ALLReduce enables collection of output from all the processors

Debug flag -> if 1, prints intermediate data to validate progress\
    To set flag to 0, for performance metrics comparison purpose