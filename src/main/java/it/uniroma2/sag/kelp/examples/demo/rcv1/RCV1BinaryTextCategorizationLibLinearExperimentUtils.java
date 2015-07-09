package it.uniroma2.sag.kelp.examples.demo.rcv1;

import java.util.List;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.LibLinearLearningAlgorithm;
import it.uniroma2.sag.kelp.utils.ExperimentUtils;
import it.uniroma2.sag.kelp.utils.evaluation.BinaryClassificationEvaluator;

public class RCV1BinaryTextCategorizationLibLinearExperimentUtils {
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

		StringLabel posLabel = new StringLabel("1");

		LearningAlgorithm learningAlgorithm = getLearningAlgorithm(c, "VEC", posLabel);
		BinaryClassificationEvaluator ev = new BinaryClassificationEvaluator(posLabel);
		List<BinaryClassificationEvaluator> nFoldCrossValidation = ExperimentUtils.nFoldCrossValidation(nfold,
				learningAlgorithm, allData, ev);

		float[] values = getValuesFrom(nFoldCrossValidation);
		float mean = it.uniroma2.sag.kelp.utils.Math.getMean(values);
		double standardDeviation = it.uniroma2.sag.kelp.utils.Math.getStandardDeviation(values);
		
		System.out.println("Accuracy mean/std on test set=" + mean + "/" + standardDeviation);	}
	
	private static float[] getValuesFrom(List<BinaryClassificationEvaluator> nFoldCrossValidation) {
		float[] ret = new float[nFoldCrossValidation.size()];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = nFoldCrossValidation.get(i).getAccuracy();
		}
		return ret;
	}

	public static LearningAlgorithm getLearningAlgorithm(float param, String representation,
			StringLabel positiveLabel) {
		LibLinearLearningAlgorithm algo = new LibLinearLearningAlgorithm(param, param, representation);
		algo.setLabel(positiveLabel);
		return algo;
	}
}
