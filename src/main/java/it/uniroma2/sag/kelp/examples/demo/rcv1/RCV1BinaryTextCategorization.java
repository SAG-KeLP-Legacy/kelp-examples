package it.uniroma2.sag.kelp.examples.demo.rcv1;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryLinearClassifier;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryMarginClassifierOutput;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.ObjectSerializer;
import it.uniroma2.sag.kelp.utils.evaluation.BinaryClassificationEvaluator;
import it.uniroma2.sag.kelp.utils.exception.NoSuchPerformanceMeasureException;

public abstract class RCV1BinaryTextCategorization {
	private StringLabel positiveLabel = new StringLabel("1");
	protected String algoSuffix = "";

	protected void foldLearn(float c, int nfold, SimpleDataset allData) {
		Dataset[] folds = allData.nFoldingClassDistributionInvariant(nfold);
		float[] accuracies = new float[folds.length];
		for (int i = 0; i < nfold; ++i) {
			SimpleDataset testSet = (SimpleDataset) folds[i];
			SimpleDataset trainingSet = getAllExcept(folds, i);
			try {
				System.out.println("start testing with C=" + c);
				accuracies[i] = test(trainingSet, c, testSet);
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
		System.out.println("Accuracy mean/std on test set=" + perf[0] + "/" + perf[1]);
	}

	private float test(SimpleDataset trainingSet, float c, SimpleDataset testSet)
			throws NoSuchPerformanceMeasureException, IOException {
		LearningAlgorithm svmSolver = getLearningAlgorithm(c, "VEC", positiveLabel);
		svmSolver.learn(trainingSet);
		BinaryLinearClassifier f = (BinaryLinearClassifier) svmSolver.getPredictionFunction();
		ObjectSerializer serializer = new JacksonSerializerWrapper();
		serializer.writeValueOnFile(svmSolver,
				"src/main/resources/rcv1/learningAlgorithmSpecification" + algoSuffix + ".klp");
		serializer.writeValueOnFile(f,
				"src/main/resources/rcv1/classificationAlgorithmSpecification" + algoSuffix + ".klp");
		BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator(positiveLabel);
		for (Example e : testSet.getExamples()) {
			BinaryMarginClassifierOutput predict = f.predict(e);
			evaluator.addCount(e, predict);
		}

		return evaluator.getPerformanceMeasure("Accuracy");
	}

	protected abstract LearningAlgorithm getLearningAlgorithm(float param, String representation,
			StringLabel positiveLabel);

	private float[] macroMeasure(float[] f1s) {
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
