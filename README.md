kelp-examples
=============


[**KeLP**][kelp-site] is the Kernel-based Learning Platform developed in the [Semantic Analytics Group][sag-site] of
the [University of Roma Tor Vergata][uniroma2-site].

This project contains some useful Java examples of the platform functionalities.
Batch learning algorithm as well as online learning algorithms usage is shown here. Different examples cover the usage of standard kernel, tree kernels and sequence kernel, with caching mechanisms.

Clone this project to obtain access to these examples by:

```
git clone https://github.com/SAG-KeLP/kelp-examples.git
```

NOTE: many of the provided examples require some memory in order to load the datasets and set up the kernel cache. You can assign memory to the Java Virtual Machine (JVM) using the option -Xmx. For instance -Xmx2G will provide 2G of memory to the JVM. In Eclipse such parameter shuld be written in Run->Run Configurations->Arguments->VM arguments .

## What you can find in the kelp-examples package

#### Classification:
* **QuestionClassification** (it.uniroma2.sag.kelp.examples.demo.qc): this class implements the Question Classification demo. It includes both kernel operating on vectors and kernel operating on trees (stk and ptk).
* **QuestionClassificationLearningFromJson** (it.uniroma2.examples.demo.qc): the same demo as QuestionClassification with the difference that the learning algorithm specification is read from a Json file.
* **RCV1BinaryTextCategorizationLibLinear**, **RCV1BinaryTextCategorizationPA** and **RCV1BinaryTextCategorizationPegasos** (it.uniroma2.sag.kelp.examples.demo.rcv1) are examples of binary classifiers on the RCV1 dataset that can be found on the LibLinear website. These classes perform a N-Fold Cross Validation and show KeLP facilities to divide a dataset in N-Fold.
* **TweetSentimentAnalysisSemeval2013** (it.uniroma2.sag.kelp.examples.demo.tweetsent2013): a demo with multiple kernels and multiple classes on a dataset on Twitter Sentiment Analysis from Semeval2013.
* **OneVsAllSVMExample** (it.uniroma2.sag.kelp.examples.main): an example that shows the usage of the OneVsAll strategy with SVM over the IRIS dataset.
* **SequenceKernelExample** (it.uniroma2.sag.kelp.examples.main): an example that shows the usage of a Sequence Kernel.
* **MultipleRepresentationExample** (it.uniroma2.sag.kelp.examples.main): a basic example showing the usage of multiple representations with multiple kernel functions with a PassiveAggressive algorithm.
* **KernelCacheExample** (it.uniroma2.sag.kelp.examples.main): an example that shows the usage of the KernelCache class to store the already computed kernel values between instances.
* **MutagClassification** 
(it.uniroma2.sag.kelp.examples.demo.mutag); an example that shows the application of graph kernels to the mutag dataset

#### Regression:
* **EpsilonSVRegressionExample** (it.uniroma2.sag.kelp.examples.demo.regression): This class contains an example of the usage of the Regression Example. The regressor implements the e-SVR learning algorithm discussed in [CC Chang & CJ Lin, 2011]. In this example a dataset is loaded from a file and then split in train and test.

#### General Purpose:
* **ClassificationDemo** (it.uniroma2.sag.kelp.examples.main): it is a meta-learner that takes in input a Json description and a dataset.




[sag-site]: http://sag.art.uniroma2.it "SAG site"
[kelp-site]: http://sag.art.uniroma2.it/demo-software/kelp/ "KeLP website"
[uniroma2-site]: http://www.uniroma2.it "University of Roma Tor Vergata"

