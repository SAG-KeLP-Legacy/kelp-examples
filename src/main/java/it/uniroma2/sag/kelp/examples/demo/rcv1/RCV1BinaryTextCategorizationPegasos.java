package it.uniroma2.sag.kelp.examples.demo.rcv1;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.pegasos.PegasosLearningAlgorithm;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryLinearClassifier;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryMarginClassifierOutput;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.ObjectSerializer;
import it.uniroma2.sag.kelp.utils.evaluation.BinaryClassificationEvaluator;
import it.uniroma2.sag.kelp.utils.exception.NoSuchPerformanceMeasureException;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

public class RCV1BinaryTextCategorizationPegasos {

	private static StringLabel positiveLabel = new StringLabel("1");

	public static void main(String[] args) {
		// Lambda = 0.001
		// Iterations = 20000
		// K = 1
		// Fold = 5
		System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "WARN");
		
		String train_file = "src/main/resources/rcv1/rcv1_train_liblsite.klp.gz";
		float[] lams = new float[]{0.001f, 0.005f, 0.1f, 0.5f, 1f};
		int nfold = 5;
		
		SimpleDataset allData = new SimpleDataset();
		try {
			allData.populate(train_file);
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println();
		float split = 0.8f;

		Dataset[] folds = allData.nFoldingClassDistributionInvariant(nfold);
		float[] accuracies = new float[folds.length];
		for (int i = 0; i < nfold; ++i) {
			SimpleDataset testSet = (SimpleDataset) folds[i];
			SimpleDataset trainingSet = getAllExcept(folds, i);
			float lam;
			try {
				lam = tune(trainingSet, split, lams);
				System.out.println("start testing with Lambda=" + lam);
				accuracies[i] = test(trainingSet, lam, testSet);
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (UnsupportedEncodingException e) {
				e.printStackTrace();
			} catch (NoSuchPerformanceMeasureException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		float[] perf = macroMeasure(accuracies);
		System.out.println("Accuracy mean/std on test set=" + perf[0] + "/"
				+ perf[1]);
	}

	private static float tune(SimpleDataset allTrainingSet, float split,
			float[] lams) throws NoSuchPerformanceMeasureException, IOException {
		if (lams.length == 1)
			return lams[0];
		float bestLam = 0.0f;
		float bestAcc = -Float.MAX_VALUE;

		Dataset[] split2 = allTrainingSet
				.splitClassDistributionInvariant(split);
		SimpleDataset trainingSet = (SimpleDataset) split2[0];
		SimpleDataset testSet = (SimpleDataset) split2[1];
		for (float lam : lams) {
			float acc = test(trainingSet, lam, testSet);
			System.out.println("LAMBDA:" + lam + "\t" + acc);
			if (acc > bestAcc) {
				bestAcc = acc;
				bestLam = lam;
			}
		}

		return bestLam;
	}

	private static float test(SimpleDataset trainingSet, float lambda,
			SimpleDataset testSet) throws NoSuchPerformanceMeasureException, IOException {
		PegasosLearningAlgorithm svmSolver = new PegasosLearningAlgorithm(1, lambda, 20000, "VEC", positiveLabel);
		svmSolver.learn(trainingSet);
		BinaryLinearClassifier f = svmSolver.getPredictionFunction();
		ObjectSerializer serializer = new JacksonSerializerWrapper();
		serializer.writeValueOnFile(svmSolver, "src/main/resources/rcv1/learningAlgorithmSpecificationPegasos.klp");
		serializer.writeValueOnFile(f, "src/main/resources/rcv1/classificationAlgorithmPegasos.klp");
		BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator(
				positiveLabel);
		for (Example e : testSet.getExamples()) {
			BinaryMarginClassifierOutput predict = f.predict(e);
			evaluator.addCount(e, predict);
		}

		return evaluator.getPerformanceMeasure("Accuracy");
	}

	private static float[] macroMeasure(float[] f1s) {
		float[] ret = new float[2];
		float sum = 0.0f;
		for (float f : f1s)
			sum += f;
		ret[0] = sum / (float) f1s.length;

		sum = 0.0f;

		for (float f : f1s)
			sum += (f - ret[0]) * (f - ret[0]);
		sum = sum / (float) f1s.length;
		ret[1] = (float) Math.sqrt(sum);

		return ret;
	}

	private static SimpleDataset getAllExcept(Dataset[] folds, int i) {
		SimpleDataset ret = new SimpleDataset();
		for (int k = 0; k < folds.length; ++k) {
			if (i != k)
				ret.addExamples(folds[k]);
		}
		return ret;
	}
}
