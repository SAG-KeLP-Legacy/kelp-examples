package it.uniroma2.sag.kelp.examples.main;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.learningalgorithm.classification.ClassificationLearningAlgorithm;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.BinaryClassificationEvaluator;
import it.uniroma2.sag.kelp.utils.evaluation.Evaluator;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

import java.io.File;
import java.util.List;

import com.fasterxml.jackson.databind.ObjectMapper;

public class ClassificationDemo {

	public static void main(String[] args) throws Exception {
		if(args.length!=3){
			System.err
			.println("Usage: trainFilePath testFilePath learningAlgorithmInputPath");
			System.exit(0);
		}

		System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "WARN");
		String trainfilePath = args[0];
		String testfilePath = args[1];
		String learningAlgoPath = args[2];

		ObjectMapper serializer = new ObjectMapper();
		ClassificationLearningAlgorithm learningAlgorithm = serializer.readValue(new File(learningAlgoPath), ClassificationLearningAlgorithm.class);
		
		System.out.println(serializer.writeValueAsString(learningAlgorithm));
	
		SimpleDataset trainDataset = new SimpleDataset();
		trainDataset.populate(trainfilePath);

		SimpleDataset testDataset = new SimpleDataset();
		testDataset.populate(testfilePath);

		// print some statistics
		System.out.println("Dataset statistics");
		System.out.print("Training Example number ");
		System.out.println(trainDataset.getNumberOfExamples());
		System.out.print("Testing Example number ");
		System.out.println(testDataset.getNumberOfExamples());

		List<Label> classes = trainDataset.getClassificationLabels();

		for (Label l : classes) {
			System.out.println("Training Label " + l.toString() + ": "
					+ trainDataset.getNumberOfPositiveExamples(l));

			System.out.println("Test Label " + l.toString() + ": "
					+ testDataset.getNumberOfPositiveExamples(l));
		}
		boolean isBinaryTask = false;
		if(classes.size()==2){
			isBinaryTask = true;
			learningAlgorithm.setLabels(classes.subList(0, 1));
		}else{
			learningAlgorithm.setLabels(classes);
		}
		
		
		learningAlgorithm.learn(trainDataset);
		Classifier classifier = learningAlgorithm.getPredictionFunction();
		Evaluator evaluator;
		if(isBinaryTask){
			evaluator = new BinaryClassificationEvaluator(learningAlgorithm.getLabels().get(0));
		}else{
			evaluator = new MulticlassClassificationEvaluator(trainDataset.getClassificationLabels());
		}
		
		for(Example ex: testDataset.getExamples()){
			ClassificationOutput prediction = classifier.predict(ex);
			evaluator.addCount(ex, prediction);
		}
		
		evaluator.compute();
		System.out.println("ACC: " + evaluator.getPerformanceMeasure("accuracy"));
		if(isBinaryTask){
			System.out.println("F1: " + evaluator.getPerformanceMeasure("F1"));
		}
		
	}
	

}
