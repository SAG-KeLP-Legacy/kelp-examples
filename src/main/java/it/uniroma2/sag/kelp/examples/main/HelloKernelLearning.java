package it.uniroma2.sag.kelp.examples.main;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.standard.PolynomialKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.passiveaggressive.KernelizedPassiveAggressiveClassification;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;

/**
 * TODO
 * 
 * Let's start with a very simple Example, that is a classification example
 * based on a linear version of the Passive Aggressive algorithm.
 * <p>
 * Dataset used are the ones used as examples in the svmlight. They have been
 * modified to be read by KeLP. In fact, a single row in KeLP must indicate what
 * kind of vectors your are using, Sparse or Dense. In the svmlight dataset
 * there are Sparse vectors, so if you open the train.dat and test.dat files you
 * can notice that each vector is enclosed in BeginVector (|BV|) and EndVector
 * (|EV|) tags.
 * <p>
 * The following example will work by adding the following Maven dependency to
 * your code:
 * <p>
 * 
 * 
 * Here you can download the converted dataset and the complete Java class. This
 * example will work by
 * <p>
 * Training set (2000 examples, 1000 of class "+1" (positive), and 1000 of class
 * "-1" (negative)) Test set (600 examples, 300 of class "+1" (positive), and
 * 300 of class "-1" (negative)) HelloLearning.java
 * <p>
 * 
 * @author Giuseppe Castellucci, Danilo Croce
 * 
 */
public class HelloKernelLearning {

	public static void main(String[] args) {
		try {
			// Read a dataset into a trainingSet variable
			SimpleDataset trainingSet = new SimpleDataset();
			trainingSet.populate("src/main/resources/hellolearning/train.klp");
			// Read a dataset into a test variable
			SimpleDataset testSet = new SimpleDataset();
			testSet.populate("src/main/resources/hellolearning/test.klp");

			// define the positive class
			StringLabel positiveClass = new StringLabel("+1");

			// print some statistics
			System.out.println("Training set statistics");
			System.out.print("Examples number ");
			System.out.println(trainingSet.getNumberOfExamples());
			System.out.print("Positive examples ");
			System.out.println(trainingSet
					.getNumberOfPositiveExamples(positiveClass));
			System.out.print("Negative examples ");
			System.out.println(trainingSet
					.getNumberOfNegativeExamples(positiveClass));

			System.out.println("Test set statistics");
			System.out.print("Examples number ");
			System.out.println(testSet.getNumberOfExamples());
			System.out.print("Positive examples ");
			System.out.println(testSet
					.getNumberOfPositiveExamples(positiveClass));
			System.out.print("Negative examples ");
			System.out.println(testSet
					.getNumberOfNegativeExamples(positiveClass));

			// instantiate a passive aggressive algorithm
			KernelizedPassiveAggressiveClassification kPA = new KernelizedPassiveAggressiveClassification();
			// indicate to the learner what is the positive class
			kPA.setLabel(positiveClass);
			// set an aggressiveness parameter
			kPA.setC(0.01f);

			// use the first (and only here) representation
			Kernel linear = new LinearKernel("0");
			// Normalize the linear kernel
			NormalizationKernel normalizedKernel = new NormalizationKernel(
					linear);
			// Apply a Polynomial kernel on the score (normalized) computed by
			// the linear kernel
			Kernel polyKernel = new PolynomialKernel(2f, normalizedKernel);
			// tell the algorithm that the kernel we want to use in learning is
			// the polynomial kernel
			kPA.setKernel(polyKernel);

			// learn and get the prediction function
			kPA.learn(trainingSet);
			Classifier f = kPA.getPredictionFunction();
			// classify examples and compute some statistics
			int correct = 0;
			for (Example e : testSet.getExamples()) {
				ClassificationOutput p = f.predict(testSet.getNextExample());
				if (p.getScore(positiveClass) > 0
						&& e.isExampleOf(positiveClass))
					correct++;
				else if (p.getScore(positiveClass) < 0
						&& !e.isExampleOf(positiveClass))
					correct++;
			}

			System.out
					.println("Accuracy: "
							+ ((float) correct / (float) testSet
									.getNumberOfExamples()));
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

}
