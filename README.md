# A-Fast-Kernel-For-GMM
The Gaussian Mixture model, which we know as GMM, is an unsupervised Machine Learning Algorithm that represents the normally distributed sub-population over a whole population. Unlike the K-means algorithm, which outputs the label of data points, GMM computes the probability of each data point and assign them into the one with the highest probability. Normally, there existing two learning models in un-supervised learning: probability model and non-probability model. Probability model represents cases where we are trying to learn P (Y |X ) instead of one single label. Within the process, we can get the probability distribution of Y from previously unknown data label X, hence outputting the continuous probability of the label and doing the soft-assignment. Non-probability model, on the other hand, is by learning a model which outputs a decision function Y = f(X), by inputting data point X, we will get one fixed result Y as the prediction, which is classified as hard assignment. The learning process of GMM including the following steps:

1. Initialize the data placement probability θ with some values(random or otherwise)
2. Model each region with a distinct distribution
3. Estimate with the soft parameter rnk using Bayes’ Rule, where we define rnk = p(zn = k|xn)
4. Solve the MLE given the soft rnk for parameter estimation purposes
5. Update θ using the rnk using MLE
6. Loop back to step 1 and compute the process again, until converges.

There are many open-source libraries and packages online that implements the process of GMM, known as EM steps. Among these libraries, Scikit-learn, which is a famous and free software machine learning library for the Python programming language, has a specific API just for GMM calculation, known as sklearn.mixture.gaussian mixture. Our goal of the project is to pick one specific API function within the sklearn.mixture.gaussian mixture and improve its performance. Upon examination, we picked the API function sklearn.mixture.gaussian mixture.estimate log gaussian prob as our project baseline function. The function takes the parameters as following: input data points X, which is a matrix with dimensions n samples by n features; means, which is an matrix with shape n components by n features; precisions chol, which array of Cholesky decomposition of the precision matrices of size n components; covariance type, which specifies the type of the covariance matrix of the Gaussian models and take input as the options: ’full’, ’tied’, ’diag’, ’spherical’. Function returns log prob, which is an matrix of the shape n samples by n components.
In the testing stage of this project, we will change the number of input data points X, n samples, with number of Gaussian components and covariance type fixed at three and spherical. This is the only thing we will be altering among the parameters. This shows how our kernel will behave with the increment of sample data size. Ideally, as the number of input data points increases, the performance compared to the naive approach will increase.

# Result 
![Result in Number of Cycles](https://github.com/cljiang74/A-Fast-Kernel-For-GMM/blob/main/images/Number_of_Cycles.png)
## Figure 1. Result Comparison in Number of Cycles between Naive and Kernel
<br/>

![Result in Speed Comparison](https://github.com/cljiang74/A-Fast-Kernel-For-GMM/blob/main/images/Speed_Performance.png)
## Figure 2. Result Comparison in Average Running Speed
<br/>

![Result in Percentage Comparison](https://github.com/cljiang74/A-Fast-Kernel-For-GMM/blob/main/images/Percentage.png)
## Figure 3. Percentage Result to Theoretical Peak
<br/>