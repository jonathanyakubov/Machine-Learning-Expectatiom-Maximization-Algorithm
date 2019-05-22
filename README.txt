README

The data provided for this assignment is points.dat. The objective of the assignment is to code the Gaussian Mixture Model using an Expectation-Maximization Algorithm. The Gaussian Mixture Model is implemented for this data because the values are continuous. The dataset consist of 2 dimensional data points. 

The Expectation-Maximization Algorithm is implemented in this assignment because this is a case of unsupervised learning and the data is non-deterministic. We cannot implement gradient descent on random variables, and thus the EM is used to optimize our parameters lambdas, mus, and sigmas. 

REQUIREMENTS: 

Please have the numpy and scipy libraries installed and imported to run this program. The use of arrays is essential in the implementation of the EM algorithm. Each iteration requires matrix multiplication and array summation. Data for this is stored in /u/cs246/data/em/. 

FUNCTIONS/IMPLEMENTATION: 

The parse_data function is used to go through the points.dat datafile and assign 90 % of the data to train data and the other 10% to the dev data. It returns two tuples, one consisting of the training data, and the other, the validation set. 

The init_model initializes the parameters lambdas, mus, and sigmas that will be used in implementation of EM for our Gaussians. There are multiple cases that need to be discussed. First, if the data is tied, there will only be one 2-by-2 sigma matrix, as the covariance will be shared among each of the clusters defined in the problem. Tied suggests that the gaussians of the data should be of the same shape. This is why the covariance matrix is shared between the clusters.The sigma 2-by-2 matrix begins as a identity matrix and multiples the diagonals by a random value between 2 and 5. The lambda initialization takes into considerations the amount of k’s requested and normalizes by that number so that the sum of all lambdas is 1. The mus are initialized by drawing a random value from a Gaussian of average 1 and STDev of 2. Through a nested loop, each mu vector of a given cluster will have random values associated with it at its initialization. In the case that the models are not tied, the sigmas still use the identity matrix for initialization with random values, but now each cluster will have its own 2 by 2 matrix with random values in the diagonal of the matrix. If the gaussian_smoketest_clusters.txt is called, then the lambdas, mus, and sigmas, will be initialized to those provided in the document. The document assumes 2 clusters for the data. The function then returns the initialized values in a tuple called model.

The third function train_model calls the initialized model and initializes several numpy matrices that will hold values in the implementation of EM. These matrices are ksi_array, ksi_k_x_n_array, and sigma_ksi_x_n_array. Through every iteration, the probability of each point in a specific gaussian will be calculated and normalized. This will be captured in the ksi_array. This is the expectation section of the algorithm that uses the initialized parameters to assign points with specific probabilities to clusters. In the maximization step, the lambdas are first calculated by summing over all the points in a specific k and dividing by the number of points. For the mu of each k, we multiply the ksi_array by each point for each given cluster and sum over that cluster to then divide by the ksi_array for that k. This yields the mu. Finally, for sigmas, if they are not tied, the product of the data point minus the mu is multiplied with the ksi_array for a given k and then summed over all points. That sum divided by the sum of the ksi_array corresponding to k column yields the new sigmas. If tied, a similar approach is implemented, but the sigma consists of the summation of each n over each k. Finally, the model parameters are returned at the end of training. 

The fourth function average_log_likelihood is used to calculate the average low likelihood of the data set given specific parameters calculated in the training function. There are multiple python exceptions implemented to catch instances where the data is tied, which means that the multivariate normal function has only 1 2-by-2 sigma matrix. The function returns the average log likelihood of the data given the model parameters 

The fifth function, extract_parameters returns the parameters that is seen after the training function is implemented. 

TESTS:

The data includes a development set and a training set. To run the algorithm without the development set and using the file, please run as follows: ./Yakubov_em_gaussian.py —-nodev —iterations ? —clusters_file gaussian_smoketest_clusters.txt —print_params. Replace the ? question mark with the iterations desired. To implement using clusters_num and using dev data, simply run ./Yakubov_em_gaussian.py —iterations ? —cluster_num ? —print_params . Replace the ? question marks with iterations and cluster numbers desired respectively. Lastly, to run using tied covariances, utilized ./Yakubov_em_gaussian.py —iterations ? —cluster_num ? —print_params —tied, where again the question marks are replaced with iterations and clusters desired, respectively. Note that when initializing from the cluster_file, you cannot also provide the cluster_num argument. Note also that —tied should not be used if initializing from a file. 

There is a commented section in the code that imports the matplotlib library to graph the validation set’s and training set’s average log likelihood with respect to iterations, to measure if there is any overfitting. More on this is provided in the discussion below. If the code is uncommented, and is run with the development data as an argument (by suppressing node), you would get similar graphs to those provided in this assignment. Please note that graphs can vary based on the initialization of parameters and convergence. EM will converge to a local optimal so it is best to vary cluster numbers and iterations to see what yields best results. More below. 

DISCUSSION: 

The Expectation-Maximization Algorithm for Gaussian Mixture Models is used to maximize the parameters lambdas, mus, and sigmas since gradient descent cannot be performed. This is because the random variables used are non-deterministic and so gradient descent cannot be directly performed on it. Nonetheless, the model parameters used can still overfit the data. Thus, it is important to check for overfitting. Using matplotlib to graph average log likelihood vs. iterations for dev data and training data, I was able to determine the best model parameters.

The first graph is using 3 clusters over 50 iterations with the sigmas NOT being tied. As observed, the training is being overfit, as the dev data is below the training average log likelihood curve. What I’ve noticed with various initializations and testing various iteration numbers and cluster numbers is that there is a tendency to converge to a average log likelihood around -4.45 for the training data and -4.55 for the dev data. Adding more clusters did not necessarily increase the average log likelihood, and the overfitting is observed substantially when observing the second graph. The best results were obtained using 3 clusters in the NO TIED case. 

This second graph uses 40 clusters over 50 iterations with NO TIED sigmas. At approximately, 23 iterations, the dev data begins to drop, while the training data shows an increase an average log likelihood. Overfitting is prevalent here. Similar experiments were run using 10 clusters with 50 iterations and NOT TIED sigmas. 

The third graph demonstrates yet again the overfitting that is occurring with the training data. As can be observed, the dev data begins to level off at iteration 10, and while it is not drastic, if the iterations would have continued, it is safe to assume that the training data would continue to level off while the dev data would decrease. 

The fourth graph demonstrates more overfitting observed at 4 clusters with 50 iterations using NO TIED sigmas. 

The fifth graph shows 2 clusters and 30 iterations with NO TIED sigmas. It seems to converge to an average log likelihood for the dev data at around -4.67. It would seem like anywhere between 5-10 iterations would be enough to get a good model before the data begins to overfit. 

In the case where the sigmas are tied, the best results observed were using 3 clusters. As seen in graph 6, when 2 clusters were used a good model is given at iterations of about 5-7, before the training data begins to overfit.The average log likelihood observed in the TIED case is roughly -5.1 for dev and -4.9 for the training data. When the cluster number was increased, it seemed that the average log likelihood increased.  Graph 7 shows this trend when using 4 clusters. The average log likelihood with 4 clusters was about -4.61 for dev data before overfitting occurred. Graph 8, using 3 clusters, shows a similar trend. Although the average log likelihood is better, the training data is leveling off, suggesting that the data is being overfit. The best results were provided with 3 clusters, as the dev data and training data are relatively the same, and overfitting is not big of a problem. 




 