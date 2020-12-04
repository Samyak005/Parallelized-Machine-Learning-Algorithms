#include <cstdio>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <math.h>
#include <omp.h>
#include <cstdlib>
#include <stdio.h>
#include <unistd.h>
#include <ctype.h>
#include <stdlib.h>
#include <memory.h>
#include <mpi.h>
using namespace std;

/*
To compile this file -> mpic++ ParallelMovieRecommendation.cpp -fopenmp
To run this program -> mpirun -host localhost -np 8 ./a.out -n 5 -p 3245

Run time options
	-np nproc : Number of parallel processors
    -i movieDataFile : training data file to build the Movie Similarity Matrix (default movies.dat)
    -r ratingsDataFile: training data file to build the Movie Similarity Matrix (default ratings.dat)
    -n top_num : number of recommendations that should be shown for a user (default 5)
    -p user   : user id for which the recommendations of movies should be shown (default value 3245)

Data of this program is the MovieLens 1M dataset from http://grouplens.org/datasets/movielens/
    movies.dat stores the Movie Id, Movie Name, Tags, for example, 9::Sudden Death (1995)::Action
    ratings.dat stores the Userid, Movie Id, Rating, Time, for example, 3123::2688::3::969323556

Collaborative Filtering
    Item-based Collaborative Filtering implementation -> computes the Movie Similarity Matrix
    In the training mode, Movie Similarity Matrix (nMovies * nMovies) is computed and stored as a file
    Cosine is used to compute the similarity score 
    O (number of users) = ratings by all users for movie m1 * ratings by all users for movie m2.
    In the testing mode, the matrix is loaded to recommend movie for users.

Notes on MPI
	More the number of processors, more is the performance speed up
    All processors are used equally to compute the Movie Similarityy Matrix (compute allocated using % mod)
    And MPI_ALLReduce enables collection of output from all the processors

Debug flag -> if 1, prints intermediate data to validate progress
    To set flag to 0, for performance metrics comparison purpose
*/

int debug = 1;
char *movieDataFile = "movies.dat";
char *ratingsDataFile = "ratings.dat";

int movieMapping[4000];
int UserNMovieRatings[6050][4000];

int nMovies = 0;
int nUsers = 0;
int nRatings = 0;

struct movie
{
    int id;
    char name[200];
    char tags[200];
};

struct movieRating
{
    int userIdentifier;
    int movieIdentifier;
    int movieRating;
};

struct movie mvData[4000];
struct movieRating ratData[1000300];

double movieSimilarity[4000][4000];
double local_sim[4000][4000];

/* Sample of Movies file
Movie Id, Movie Name, Tags
9::Sudden Death (1995)::Action
10::GoldenEye (1995)::Action|Adventure|Thriller
11::American President, The (1995)::Comedy|Drama|Romance
12::Dracula: Dead and Loving It (1995)::Comedy|Horror
13::Balto (1995)::Animation|Children's
14::Nixon (1995)::Drama
15::Cutthroat Island (1995)::Action|Adventure|Romance
*/

int read_movies()
{
    // input file stream (inputFileStrm) for reading data from file
    char readLine[256];
    ifstream movieFile(movieDataFile, ios::in);
    int flag = 0;
    if (!movieFile.is_open())
    {
        printf("file open failed");
        return -1;
    }
    // read from inputFileStrm into string 'readLine'
    nMovies = 0;
    while (movieFile.getline(readLine, 255))
    {
        sscanf(readLine, "%d::%[^::]::%s", &mvData[nMovies].id, mvData[nMovies].name, mvData[nMovies].tags);
        movieMapping[mvData[nMovies].id] = nMovies;
        ++nMovies;
    }
    if (debug)
        printf("number of movies read %d \n", nMovies);
    movieFile.close();
    return 1;
}

/* Sample of ratings file
Userid, Movie Id, Rating, Time
3123::2688::3::969323556
3123::3638::5::969324299
3123::2694::4::969323705
3123::2699::3::969323685
3123::2840::4::969323955
3123::2841::5::969323955
3123::3658::5::969324132
*/

int read_ratings()
{
    // input file stream (inputFileStrm) for reading data from file
    char readLine[256];
    ifstream ratingsFile(ratingsDataFile, ios::in);
    if (!ratingsFile.is_open())
    {
        printf("file open failed");
        return -1;
    }
    // read from inputFileStrm into string 'readLine'
    int lastID = -1;
    nRatings = 0;
    nUsers = 0;
    int time;
    while (ratingsFile.getline(readLine, 255))
    {
        sscanf(readLine, "%d::%d::%d::%d", &ratData[nUsers].userIdentifier, &ratData[nUsers].movieIdentifier, &ratData[nUsers].movieRating, &time);
        if (lastID != ratData[nUsers].userIdentifier)
        {
            lastID = ratData[nUsers].userIdentifier;
            ++nUsers;
        }
        ++nRatings;
    }
    if (debug)
        printf("number of ratings read %d number of Users %d  \n", nRatings, nUsers);
    ratingsFile.close();
    return 1;
}

void update_movieNuser()
{
    for (int i = 0; i < nRatings; ++i)
    {
        int userdim = ratData[i].userIdentifier - 1;
        int moviedim = movieMapping[ratData[i].movieIdentifier];
        UserNMovieRatings[userdim][moviedim] = ratData[i].movieRating;
    }
}

// Compute Cosine between two movie vectors
double computeMovieSimilarity(int m1, int m2)
{
    int numR = 0;
    double dot = 0;
    double denom_x = 0;
    double denom_y = 0;

    //printf("in computeMovieSimilarity %d %d\n", m1, m2);
    for (int i = 0; i < nUsers; ++i)
    {
        // only compute for positive values (ratings are always positive)
        // ignore zero values that is missing movieRating from the user 'i' for the movie m1 or m2
        if ((UserNMovieRatings[i][m1] > 0) && (UserNMovieRatings[i][m2] > 0))
        {
            numR++;
            dot += UserNMovieRatings[i][m1] * UserNMovieRatings[i][m2];
            denom_x += UserNMovieRatings[i][m1] ^ 2;
            denom_y += UserNMovieRatings[i][m2] ^ 2;
        }
    }

    if ((numR <= 1) || (denom_x == 0) || (denom_y == 0))
    {
        return 0.0;
    }
    return dot / (sqrt(denom_x) * sqrt(denom_y));
}

//Calculate similarity of all movies. MPI Allgatherv is applied to achieve higher performance.
void movie_similarity()
{
    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if ((debug) && (rank == 0))
        printf("number of processors %d\n", nproc);

    int *count = new int[nproc];
    for (int m1 = 0; m1 < nMovies; ++m1)
    {
        for (int m2 = 0; m2 < nMovies; ++m2)
        {
            local_sim[m1][m2] = 0.0;
        }
    }

    if (nproc != 1)
    {
        for (int m1 = 0; m1 < nMovies; ++m1)
        {
            for (int m2 = 0; m2 < m1; ++m2)
            {
                // allocate to different processes as per rank
                if ((m1 + m2) % nproc == rank)
                {
                    local_sim[m2][m1] = computeMovieSimilarity(m1, m2);
                    count[rank]++;
                }
            }
        }
        if (debug)
        {
            printf("rank %d count %d 2\n", rank, count[rank]);
        }

        // sum all data in local_sim in movieSimilarity
        MPI_Allreduce(local_sim[0], movieSimilarity[0], nMovies * nMovies, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // use the symmetric property to complete the matrix
        for (int m1 = 0; m1 < nMovies; ++m1)
        {
            for (int m2 = 0; m2 < m1; ++m2)
            {
                movieSimilarity[m1][m2] = movieSimilarity[m2][m1];
            }
        }
    }
    else
    {
        int count = 0;
        for (int m1 = 0; m1 < nMovies; ++m1)
        {
            for (int m2 = 0; m2 < m1; ++m2)
            {
                movieSimilarity[m2][m1] = computeMovieSimilarity(m1, m2);
                movieSimilarity[m1][m2] = movieSimilarity[m2][m1];
                count++;
            }
        }
        if (debug)
            printf("rank %d count %d\n", rank, count);
    }
}

int *bbsort(int top_num, int *movieIdentifier, double *score)
{
    int maxIndex;
    int i;
    int j;
    int tempMovieId;
    int tempScore;
    int count = 3883;
    int *r = new int[top_num];
    double max_score;

    for (i = 1; i <= top_num; i++)
    {
        maxIndex = -1;
        max_score = -1000000;
        for (j = 0; j <= (count - i - 1); j++)
        {
            if (score[j] > max_score)
            {
                max_score = score[j];
                maxIndex = j;
            }
        }
        // swap max_index with j
        tempMovieId = movieIdentifier[j];
        tempScore = score[j];
        movieIdentifier[j] = movieIdentifier[maxIndex];
        score[j] = score[maxIndex];
        movieIdentifier[maxIndex] = tempMovieId;
        score[maxIndex] = tempScore;

        //printf("sorted: i %d, score %f max score %f movie id %d\n", count-i, ranking_info[j].score, max_score, ranking_info[j].movieIdentifier);
        r[i - 1] = movieIdentifier[j];
    }
    return r;
}

//Command line arguments: -m mode -n top_num -p user -i movieinput -r ratinginput
int main(int argc, char *argv[])
{
    int opt;
    extern char *optarg;
    int is_output_timing = 1;

    int nproc, rank;
    double init_time, training_time;

    omp_set_num_threads(8);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int *result;
    int user, top_num, mode;
    int uid_num;

    top_num = 5;
    user = 3245;
    while ((opt = getopt(argc, argv, "i:m:n:p:r")) != EOF)
    {
        switch (opt)
        {
        case 'i':
            movieDataFile = optarg;
            break;
        case 'n':
            top_num = atoi(optarg);
            break;
        case 'p':
            user = atoi(optarg);
            break;
        case 'r':
            ratingsDataFile = optarg;
            break;
        default:
            break;
        }
    }

    // Read data from Movies and Ratings file
    read_movies();
    read_ratings();
    update_movieNuser();

    init_time = MPI_Wtime();
    movie_similarity();
    training_time = MPI_Wtime() - init_time;

    if (is_output_timing)
    {
        printf("rank %d in output time computation %f\n", rank, training_time);
    }

    if (rank == 0)
    {
        int movieIdentifier[nMovies];
        double score[nMovies];
        for (int m1 = 0; m1 < nMovies; ++m1)
        {
            movieIdentifier[m1] = mvData[m1].id;
            score[m1] = 0;
            for (int m2 = 0; m2 < nMovies; ++m2)
            {
                score[m1] += (UserNMovieRatings[user - 1][m2] * movieSimilarity[m1][m2]);
            }
        }
        result = bbsort(top_num, movieIdentifier, score);
        printf("Recommendation for user %d\n", user);
        printf(" Rank | Movie Name\n");
        for (int m1 = 0; m1 < top_num; ++m1)
        {
            printf(" %4d | %s\n", m1 + 1, mvData[movieMapping[result[m1]]].name);
        }
    }

    MPI_Finalize();
    return 0;
}
