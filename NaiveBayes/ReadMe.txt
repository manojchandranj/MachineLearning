Reuters-21578 dataset, consists of documents that were appeared in Reuter’s newswire in 1987. Each document was then manually categorized into a topic among over 100 topics. Only the documents on “earn” and “acquisition” (acq) topics are used, (documents assigned to topics other than "earn" or "acq" are not in the dataset). As features, the frequency (counts) of each word occurred in the document are used. This model is known as bag of words model and it is frequently used in text categorization.

Matlab version of the dataset in .mat format is used, (reuters.mat) so that it can be directly imported into Matlab/Octave.

Naive Bayes algorithm described in Tom Mitchell's book is implemented. And the following paper was referred for more details: http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf

To avoid zero probabilities, one laplace smoothing was added and made sure that all the calculations are done in log-scale to avoid underflow (see the above mentioned paper).

The algorithm is made to learn from the training set and report the accuracy on the test set.

