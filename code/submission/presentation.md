In this project, I recognised several challenges that needed to be overcome
the sparsity of content I.e only a small subset of features is relevant per user, and there is possibly missing information that needs to be inferred.
the opaque nature of the data where intuition is limited.
there isnt a lot of data
I addressed this problem in several axes:
statistics - I used a statistical(baysian) method to compute the probability a person is a "1", using real world events. I.e I assumed users that are apart of the same community will go on thr same websites around the same time - which is why I binned the time series to 6 hour bins, applied a gaussian filter, and computed the probability of a user being 1, given they were active In that bin. for that I used statistics from the training data.
I used a power spectrum over the general activity of a user in order to get temporal patterns, instead of multiple temporal features e.g time of day of activity .
I applied this method indetail to the domains, but in an aggregated form to the urls, in order to avoid too many features.
Didnt get the chance to submit , but I used node2vec in order to simulate random walks and derive meaningful url clusters, and augment users with random walks.
validation:
my constant validation was "how do people with fewer features perform" which is why I used weighted sampling in the model.