package it.uniroma2.sag.kelp.examples.demo.qc;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

import java.io.File;
import java.util.List;

public class QuestionClassificationLearningFromJson {

	public static void main(String[] args) {
		try {
			System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "WARN");
			// Read a dataset into a trainingSet variable
			SimpleDataset trainingSet = new SimpleDataset();
			trainingSet.populate("src/main/resources/qc/train_5500.coarse.klp.gz");

			SimpleDataset testSet = new SimpleDataset();
			testSet.populate("src/main/resources/qc/TREC_10.coarse.klp.gz");

			// print some statistics
			System.out.println("Training set statistics");
			System.out.print("Examples number ");
			System.out.println(trainingSet.getNumberOfExamples());

			List<Label> classes = trainingSet.getClassificationLabels();

			for (Label l : classes) {
				System.out.println("Training Label " + l.toString() + " " + trainingSet.getNumberOfPositiveExamples(l));
				System.out.println("Training Label " + l.toString() + " " + trainingSet.getNumberOfNegativeExamples(l));

				System.out.println("Test Label " + l.toString() + " " + testSet.getNumberOfPositiveExamples(l));
				System.out.println("Test Label " + l.toString() + " " + testSet.getNumberOfNegativeExamples(l));
			}

			JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();
			OneVsAllLearning ovaLearner = serializer.readValue(
					new File("src/main/resources/qc/learningAlgorithmSpecification.klp"), OneVsAllLearning.class);
			
			ovaLearner.setLabels(classes);

			// learn and get the prediction function
			ovaLearner.learn(trainingSet);
			Classifier f = ovaLearner.getPredictionFunction();

			// classify examples and compute some statistics
			MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(classes);
			for (Example e : testSet.getExamples()) {
				ClassificationOutput p = f.predict(testSet.getNextExample());
				evaluator.addCount(e, p);
			}

			System.out.println("Accuracy: " + evaluator.getAccuracy());
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}
}
