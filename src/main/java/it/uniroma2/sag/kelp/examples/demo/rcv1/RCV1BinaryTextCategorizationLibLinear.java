package it.uniroma2.sag.kelp.examples.demo.rcv1;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.LibLinearLearningAlgorithm;

public class RCV1BinaryTextCategorizationLibLinear extends RCV1BinaryTextCategorization {
	protected String algoSuffix = "LibLinear";
	
	public static void main(String[] args) {
		System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "WARN");

		String train_file = "src/main/resources/rcv1/rcv1_train_liblsite.klp.gz";
		float c = 1f;
		int nfold = 5;

		SimpleDataset allData = new SimpleDataset();
		try {
			allData.populate(train_file);
		} catch (Exception e) {
			e.printStackTrace();
		}

		RCV1BinaryTextCategorization foldLearning = new RCV1BinaryTextCategorizationLibLinear();
		foldLearning.foldLearn(c, nfold, allData);
	}
	
	@Override
	protected LearningAlgorithm getLearningAlgorithm(float param, String representation, StringLabel positiveLabel) {
		LibLinearLearningAlgorithm algo = new LibLinearLearningAlgorithm(param,param,representation);
		algo.setLabel(positiveLabel);
		
		return algo;
	}
}
