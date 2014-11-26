package it.uniroma2.sag.kelp.examples.main;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.passiveaggressive.LinearPassiveAggressiveClassification;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;

public class HelloLearning {
	
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
			System.out.println(trainingSet.getNumberOfPositiveExamples(positiveClass));
			System.out.print("Negative examples ");
			System.out.println(trainingSet.getNumberOfNegativeExamples(positiveClass));
			
			System.out.println("Test set statistics");
			System.out.print("Examples number ");
			System.out.println(testSet.getNumberOfExamples());
			System.out.print("Positive examples ");
			System.out.println(testSet.getNumberOfPositiveExamples(positiveClass));
			System.out.print("Negative examples ");
			System.out.println(testSet.getNumberOfNegativeExamples(positiveClass));
			
			// instantiate a passive aggressive algorithm
			LinearPassiveAggressiveClassification passiveAggressiveAlgorithm = new LinearPassiveAggressiveClassification();
			// use the first (and only here) representation
			passiveAggressiveAlgorithm.setRepresentation("0");
			// indicate to the learner what is the positive class
			passiveAggressiveAlgorithm.setLabel(positiveClass);
			// set an aggressiveness parameter
			passiveAggressiveAlgorithm.setC(0.01f);

			// learn and get the prediction function
			passiveAggressiveAlgorithm.learn(trainingSet);
			Classifier f = passiveAggressiveAlgorithm.getPredictionFunction();
			// classify examples and compute some statistics
			int correct=0;
			for (Example e : testSet.getExamples()) {
				ClassificationOutput p = f.predict(testSet.getNextExample());
				if (p.getScore(positiveClass) > 0 && e.isExampleOf(positiveClass))
					correct++;
				else if (p.getScore(positiveClass) < 0 && !e.isExampleOf(positiveClass))
					correct++;
			}
			
			System.out.println("Accuracy: " + ((float)correct/(float)testSet.getNumberOfExamples()));
		} catch (Exception e1) {
			e1.printStackTrace();
		}	
	}	

}
