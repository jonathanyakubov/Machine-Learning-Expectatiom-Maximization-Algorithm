#!/usr/local/bin/python3
import numpy as np
if not __file__.endswith('_em_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as LastName_em_gaussian.py (replacing LastName with your last name)!')
    exit(1)

DATA_PATH = "/Users/jonathanyakubov/Desktop/MachineLearning/MLhw7/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    clusters = []
    if args.cluster_num:
    	lambdas = np.zeros(args.cluster_num)
    	numbers_clusters=int(len(lambdas))
    	mus = np.zeros((args.cluster_num,2))
    	# print(mus)
    	if not args.tied:
    		sigmas=np.zeros((args.cluster_num,2,2))
    		for i in range(len(sigmas)):
    			sigmas[i]=np.identity(2)
    			rand_integer=np.random.randint(1,10)
    			sigmas[i][0][0]=rand_integer
    			sigmas[i][1][1]=rand_integer
    		# print(sigmas)
    	else:
    		sigmas=np.zeros((2,2))
    		sigmas=np.identity(2)
    		rand_integer=np.random.randint(2,5)
    		sigmas[0][0]=rand_integer
    		sigmas[1][1]=rand_integer
    		# print(sigmas)
    	for i in range(numbers_clusters):
    		for n in range(2):
    			mus[i][n]=np.random.normal(loc=1,scale=2)
    	for i in range(len(lambdas)):       #initialization lambdas 
        	lambdas[i]=1/int(len(lambdas))
       #  TODO: randomly initialize clusters (lambdas, mus, and sigmas)
       #  raise NotImplementedError #remove when random initialization is implemented
    else:
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float,line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)

    #TODO: do whatever you want to pack the lambdas, mus, and sigmas into the model variable (just a tuple, or a class, etc.)
    #NOTE: if args.tied was provided, sigmas will have a different shape
    model = (lambdas, mus,sigmas)
    # print(model)
    # raise NotImplementedError #remove when model initialization is implemented
    return model

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
   #  from matplotlib import pyplot as plt
    lambdas, mus, sigmas=model
#     print(mus)
#     print(lambdas)
    ksi_array=np.zeros((len(train_xs),len(lambdas)))  #initialization of ksi array n by k
    ksi_k_x_n_array=np.zeros((len(train_xs),len(lambdas)),dtype=object)   #array initialized for mu calculations
    sigma_ksi_x_n_array=np.zeros((len(train_xs),len(lambdas)),dtype=object)  #array initialized for sigma calculations
#     dev_likelihoods=[] #initialization of dev likelihoods
#     train_likelihoods=[] #initialization of train  likelihoods
#     iterations=[]   #initialization of iterations
    
    for p in range(args.iterations):
    	# iterations.append(p+1)
    	# print(p+1)
    
    	for i in range(len(train_xs)):
    		sum_=0
    		for k in range(len(lambdas)):  #just k 
    			if args.tied:
    				N_k_n=multivariate_normal.pdf(train_xs[i],mean=mus[k],cov=sigmas) #this is an array or scalar?
    			else:
    				N_k_n=multivariate_normal.pdf(train_xs[i],mean=mus[k],cov=sigmas[k])
    			P_k_n=lambdas[k]*N_k_n  #unnormalized probability 
    			sum_+=P_k_n	     #sum them 
    		for c in range(len(lambdas)):  #c is another k
    			if args.tied:
    				N_k_n=multivariate_normal.pdf(train_xs[i],mean=mus[c],cov=sigmas)
    			else:
    				N_k_n=multivariate_normal.pdf(train_xs[i],mean=mus[c],cov=sigmas[c]) #this is an array or scalar?
    			P_k_n=lambdas[c]*N_k_n
    			ksi_k_n=P_k_n/sum_  #the ksi values 
    			ksi_array[i][c]=ksi_k_n
    	# print(ksi_array)
    	for l in range(len(lambdas)):  #l is another k 
    		column_k=ksi_array[:,l]
    		# print(len(column_k))
    		sum_ksi_column_k=np.sum(column_k)
    		lambdas[l]=(sum_ksi_column_k)/(len(train_xs))
    		# print(lambdas)
    		
    	
    		for i in range(len(train_xs)):
    			
    			ksi_k_x_n_array[i][l]=ksi_array[i][l]*train_xs[i] #scalar times datapoint with 2D's
    		# print(ksi_k_x_n_array)
    		column_k_n_x_n=ksi_k_x_n_array[:,l]
#     		print(column_k_n_x_n)
    		# print((np.sum(column_k_n_x_n,axis=0))/(sum_ksi_column_k)) 
    		mus[l]=(np.sum(column_k_n_x_n,axis=0))/(sum_ksi_column_k) 
    		# print(mus[l])
    	
    	
    		if not args.tied:
    			for n in range(len(train_xs)):
    				x=np.reshape(train_xs[n], (2,1))
    		# 		print(x)
    				mu=np.reshape(mus[l],(2,1))
    				# print(mu)
    				x_n_mu_difference=x-mu
    				# x_n_mu_difference=train_xs[n]-mus[l]
    				# print(x_n_mu_difference)
    				product=np.matmul(x_n_mu_difference,x_n_mu_difference.T)
    				# print(product)
    				sigma_ksi_x_n_array[n][l]=ksi_array[n][l]*product
    			
    			# print(sigma_ksi_x_n_array)
    			column_k_sigma_x_n=sigma_ksi_x_n_array[:,l]
    			# print(column_k_sigma_x_n)
#     			print((np.sum(column_k_sigma_x_n,axis=0))/(sum_ksi_column_k))
    			sigmas[l]=(np.sum(column_k_sigma_x_n))/(sum_ksi_column_k)  #might be axis =0.. need to check
    			# print(sigmas[l])
#     			print(sigmas[l].shape)
    	if args.tied:
    		# sigmas = np.zeros((2,2))  #i'm going to need to change this once initialized 
    		# print(sigmas)
    		for j in range(len(train_xs)):
    			for d in range(len(lambdas)):
    				x=np.reshape(train_xs[j], (2,1))
    				mu=np.reshape(mus[d],(2,1))
    				x_n_mu_difference=x-mu
    				sigmas+=ksi_array[j][d]*np.matmul(x_n_mu_difference,x_n_mu_difference.T)
    				# print(sigmas)
    		sigmas=(sigmas/len(train_xs))  #the sigma in the case of tied 
    		# print(sigmas)
    	model=(lambdas,mus,sigmas)
    	# print(model)
#     	if not args.nodev:
#     		dev_likelihood=average_log_likelihood(model, dev_xs)
#     		dev_likelihoods.append(dev_likelihood)
#     		train_likelihood=average_log_likelihood(model, train_xs)
#     		train_likelihoods.append(train_likelihood)
#     if not args.nodev:
#     	plt.plot(iterations,train_likelihoods, label="Training Data Avg Log Likelihood")
#     	plt.plot(iterations,dev_likelihoods, label="Dev Data Avg Log Likelihood")
#     	plt.xlabel("Iterations")
#     	plt.ylabel("Average Log Likelihood")
#     	plt.title("Average Log Likelihood vs. Iterations")
#     	plt.legend()
#     	plt.show()
    
    	
    #NOTE: you can use multivariate_normal like this:
    #probability_of_xn_given_mu_and_sigma = multivariate_normal(mean=mu, cov=sigma).pdf(xn)
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
   #  raise NotImplementedError #remove when model training is implemented
    return model

def average_log_likelihood(model, data):
    from math import log
    from scipy.stats import multivariate_normal
    
    lambdas,mus,sigmas=model
    max_likeli=0
    try:
    	for n in range(len(data)):
    		N_k_n=0
    		for k in range(len(lambdas)):
    			N_k_n+=(lambdas[k]*multivariate_normal.pdf(data[n],mean=mus[k],cov=sigmas[k]))

    		max_likeli+=log(N_k_n)
    except ValueError:    #captures instances where tied 
    	for n in range(len(data)):
    		N_k_n=0
    		for k in range(len(lambdas)):
    			N_k_n+=(lambdas[k]*multivariate_normal.pdf(data[n],mean=mus[k],cov=sigmas)) 
    		max_likeli+=log(N_k_n)
    except IndexError:   #captures instances where tied 
    	for n in range(len(data)):
    		N_k_n=0
    		for k in range(len(lambdas)):
    			N_k_n+=(lambdas[k]*multivariate_normal.pdf(data[n],mean=mus[k],cov=sigmas))
    	
    		max_likeli+=log(N_k_n)
    ll=max_likeli/len(data)
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
   
    # raise NotImplementedError #remove when average log likelihood calculation is implemented
    return ll

def extract_parameters(model):
    #TODO: extract lambdas, mus, and sigmas from the model and return them (same type and shape as in init_model)
    lambdas = model[0]
    mus = model[1]
    sigmas = model[2]
    # raise NotImplementedError #remove when parameter extraction is implemented
    return lambdas, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    ll_train = average_log_likelihood(model, train_xs)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()