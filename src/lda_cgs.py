import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
import time

### Desired extensions:
# show prob(w|z) in get_top_words_for_topics()
# create horizontal bar chart (for p(w|z)) in get_top_words_for_topics(), one column for each topic
# code perplexity/likelihood to help evaluate when to stop iterating, get perplexity score of train, test sets
# transform() for a test set

class LDA_CGS():
    def __init__(self, K = 40, document_topic_prior = None, topic_word_prior = 0.01,
                 num_iterations = 100, evaluate_every = 10, perp_tol = 1e-1, verbose = True):
        self.K = K
        if document_topic_prior == None:
            self.document_topic_prior = 50/self.K
        else:
            self.document_topic_prior = document_topic_prior
        self.topic_word_prior = topic_word_prior
        self.num_iterations = num_iterations
        self.evaluate_every = evaluate_every
        self.perp_tol = perp_tol
        self.verbose = verbose

    def init_Z_and_counters(self):
        # initialize Z, sigma, delta
        self.Z = []
        self.indexes = []
        self.sigma_ = np.zeros((self.M, self.K)) #+ self.document_topic_prior
        self.delta_ = np.zeros((self.K, self.V)) #+ self.topic_word_prior
        self.delta_z_ = np.zeros(self.K) #+ self.topic_word_prior*self.V

        if self.verbose:
            print("Initializing variables...")
        # for every document,
        for j in range(self.M):
            Z_j = []
            indexes_j = []
            # for every word in document,
            for i in np.nonzero(self.X[j,:])[1]:
                # for every occurrence of that word,
                for _ in range(self.X[j,i]):
                    # randomly sample z from topics
                    z = np.random.randint(0, self.K)

#                     # sample z from posterior
#                     p_z = np.multiply(self.sigma_[j, :],
#                                       self.delta_[:, i]/self.delta_z_)
#                     z = np.random.choice(a = range(self.K), p = p_z/sum(p_z), size = 1)

                    # append token to list of tokens for that document
                    Z_j.append(z)
                    # append index of the word list of indexes for that document
                    indexes_j.append(i)
                    # update counters
                    self.sigma_[j,z] += 1
                    self.delta_[z,i] += 1
                    self.delta_z_[z] += 1
            self.Z.append(Z_j)
            self.indexes.append(indexes_j)

    def perform_gibbs_sampling(self):
        if self.verbose:
            print("Performing Gibbs Sampling...")
            start_time = time.time()
        # for each iteration,
        for epoch in range(1, self.num_iterations + 1):
            # for each document,
            for m in range(self.M):
                # for each word in document,
                for iterator in range(len(self.Z[m])):
                    # get sampled topic of word
                    z_mn = self.Z[m][iterator]
                    # get vocabulary index of word
                    vocab_index = self.indexes[m][iterator]

                    # decrement counters
                    self.sigma_[m, z_mn] -= 1
                    self.delta_[z_mn, vocab_index] -= 1
                    self.delta_z_[z_mn] -= 1

                    # calculate P(z_mn|Z_-mn, alpha, beta)
                    P_z_mn = np.multiply((self.sigma_[m, :] + self.document_topic_prior),
                                         (self.delta_[:, vocab_index] + self.topic_word_prior)/(self.delta_z_ + self.V*self.topic_word_prior))
#                     P_z_mn = np.multiply((self.sigma_[m, :]),
#                                          (self.delta_[:, vocab_index])/(self.delta_z_))

                    # sample new z_mn from P(z_mn|Z_-mn, alpha, beta)
                    new_z_mn = np.random.choice(a = range(self.K), p = P_z_mn/sum(P_z_mn), size = 1)

                    # increment counters
                    self.sigma_[m, new_z_mn] += 1
                    self.delta_[new_z_mn, vocab_index] += 1
                    self.delta_z_[new_z_mn] += 1

                    # update z_mn
                    self.Z[m][iterator] = new_z_mn
            # print progress after every 10% interval
            if self.verbose:
                # if epoch%(self.num_iterations/10) == 0:
                    # print("\tIteration %d, %.2f%% complete, %0.0f mins elapsed"%(epoch,
                    #                                                            100*epoch/self.num_iterations,
                    #                                                            (time.time() - start_time)/60))
                print("\tIteration %d, %.2f%% complete, %0.0f mins elapsed"%(epoch,
                                                                           100*epoch/self.num_iterations,
                                                                           (time.time() - start_time)/60))
#                     print("\tPerplexity: %.2f"%self.perplexity())
        self.calculate_document_topic_matrix()
        self.calculate_topic_word_matrix()
        if self.verbose:
            print("Total time elapsed: %0.0f mins"%((time.time() - start_time)/60))

    def fit(self, X):
        self.M, self.V = X.shape
        self.N = np.sum(X)
        self.X = X
        self.init_Z_and_counters()
        self.perform_gibbs_sampling()

    def calculate_document_topic_matrix(self):
        self.document_topic_matrix_ = (self.sigma_ + self.document_topic_prior)/((np.sum(self.sigma_, axis = 1) + self.K*self.document_topic_prior)[:, np.newaxis])

    def calculate_topic_word_matrix(self):
        self.topic_word_matrix_ = (self.delta_ + self.topic_word_prior)/((np.sum(self.delta_, axis = 1) + self.V*self.topic_word_prior)[:, np.newaxis])

#     def perplexity(self):
#         self.calculate_document_topic_matrix()
#         self.calculate_topic_word_matrix()
#         log_sum = 0
#         for m in range(self.\M):
#             for n in range(self.V):
#                 sum = 0
#                 for k in range(self.K):
#                     sum += (self.document_topic_matrix_[m,k] * self.topic_word_matrix_[k,n])
#                 log_sum += np.log(sum)
#         return(np.exp(-log_sum/self.N))


#     def transform():
