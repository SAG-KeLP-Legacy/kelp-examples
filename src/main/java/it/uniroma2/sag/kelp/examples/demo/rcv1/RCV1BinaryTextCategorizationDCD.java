package it.uniroma2.sag.kelp.examples.demo.rcv1;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.dcd.DCDLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.dcd.DCDLoss;

public class RCV1BinaryTextCategorizationDCD extends RCV1BinaryTextCategorization {
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

		RCV1BinaryTextCategorization foldLearning = new RCV1BinaryTextCategorizationDCD();
		foldLearning.foldLearn(c, nfold, allData);
	}
	
	@Override
	protected LearningAlgorithm getLearningAlgorithm(float param, String representation, StringLabel positiveLabel) {		
		/**
		 * The considered Loss function (L1 or L2)
		 */
		DCDLoss dcdLoss = DCDLoss.L2;
		/**
		 * This boolean parameter determines the use of bias <code>b</code> in the
		 * classification function <cod>f(x)=wx+b</code>. If usebias is set to
		 * <code>false</code> the bias is set to 0.
		 */
		boolean usebias = true;
		/**
		 * The number of iteration of the main algorithm
		 */
		int iterations = 20;

		DCDLearningAlgorithm algo = new DCDLearningAlgorithm(param, param,
				dcdLoss, usebias, iterations, representation);
		algo.setLabel(positiveLabel);
		
		return algo;
	}
}
