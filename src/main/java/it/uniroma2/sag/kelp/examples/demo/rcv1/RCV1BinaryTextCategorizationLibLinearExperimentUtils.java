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

		float[] m = macroMeasure(nFoldCrossValidation);

		System.out.println("Accuracy mean/std on test set=" + m[0] + " - " + m[1]);
	}

	private static float[] macroMeasure(List<BinaryClassificationEvaluator> nFoldCrossValidation) {
		float[] ret = new float[2];
		float sum = 0.0f;
		for (BinaryClassificationEvaluator f : nFoldCrossValidation)
			sum += f.getF1();
		ret[0] = sum / (float) nFoldCrossValidation.size();

		sum = 0.0f;

		for (BinaryClassificationEvaluator f : nFoldCrossValidation)
			sum += (f.getF1() - ret[0]) * (f.getF1() - ret[0]);
		sum = sum / (float) nFoldCrossValidation.size();
		ret[1] = (float) Math.sqrt(sum);

		return ret;
	}

	public static LearningAlgorithm getLearningAlgorithm(float param, String representation,
			StringLabel positiveLabel) {
		LibLinearLearningAlgorithm algo = new LibLinearLearningAlgorithm(param, param, representation);
		algo.setLabel(positiveLabel);
		return algo;
	}
}
