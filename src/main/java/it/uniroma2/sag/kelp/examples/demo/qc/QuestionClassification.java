package it.uniroma2.sag.kelp.examples.demo.qc;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexKernelCache;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexSquaredNormCache;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.tree.PartialTreeKernel;
import it.uniroma2.sag.kelp.kernel.tree.SubTreeKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

import java.util.List;

public class QuestionClassification {

	public static void main(String[] args) {
		try {
			System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "WARN");
			// Read a dataset into a trainingSet variable
			SimpleDataset trainingSet = new SimpleDataset();
			trainingSet
					.populate("src/main/resources/qc/train_5500.coarse.klp.gz");

			SimpleDataset testSet = new SimpleDataset();
			testSet.populate("src/main/resources/qc/TREC_10.coarse.klp.gz");

			String tkString = "bow";

			// print some statistics
			System.out.println("Training set statistics");
			System.out.print("Examples number ");
			System.out.println(trainingSet.getNumberOfExamples());

			// Set the cache size
			int cacheSize = trainingSet.getNumberOfExamples()
					+ testSet.getNumberOfExamples();

			List<Label> classes = trainingSet.getClassificationLabels();

			for (Label l : classes) {
				System.out.println("Training Label " + l.toString() + " "
						+ trainingSet.getNumberOfPositiveExamples(l));
				System.out.println("Training Label " + l.toString() + " "
						+ trainingSet.getNumberOfNegativeExamples(l));

				System.out.println("Test Label " + l.toString() + " "
						+ testSet.getNumberOfPositiveExamples(l));
				System.out.println("Test Label " + l.toString() + " "
						+ testSet.getNumberOfNegativeExamples(l));
			}

			// Initialize the kernel function
			Kernel usedKernel = null;
			if (tkString.equalsIgnoreCase("stk")) {
				// Kernel for the first representation (0-index)
				Kernel stkgrct = new SubTreeKernel(0.4f, "grct");
				stkgrct.setSquaredNormCache(new FixIndexSquaredNormCache(
						cacheSize));
				Kernel normPtkGrct = new NormalizationKernel(stkgrct);
				usedKernel = normPtkGrct;
			} else if (tkString.equalsIgnoreCase("bow")) {
				Kernel linear = new LinearKernel("bow");
				linear.setSquaredNormCache(new FixIndexSquaredNormCache(
						cacheSize));
				linear = new NormalizationKernel(linear);

				usedKernel = linear;
			} else if (tkString.equalsIgnoreCase("ptk")) {
				// Kernel for the first representation (0-index)
				Kernel ptkgrct = new PartialTreeKernel(0.4f, 0.4f, 5f, "grct");
				ptkgrct.setSquaredNormCache(new FixIndexSquaredNormCache(
						cacheSize));
				Kernel normPtkGrct = new NormalizationKernel(ptkgrct);
				usedKernel = normPtkGrct;
			}
			// Set cache to the kernel
			usedKernel.setKernelCache(new FixIndexKernelCache(cacheSize));

			JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();

			// instantiate an svmsolver
			BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
			svmSolver.setKernel(usedKernel);
			svmSolver.setCn(3);
			svmSolver.setFairness(true);

			OneVsAllLearning ovaLearner = new OneVsAllLearning();
			ovaLearner.setBaseAlgorithm(svmSolver);
			ovaLearner.setLabels(classes);
			serializer.writeValueOnFile(ovaLearner,
					"src/main/resources/qc/learningAlgorithmSpecification.klp");

			// learn and get the prediction function
			ovaLearner.learn(trainingSet);
			Classifier f = ovaLearner.getPredictionFunction();
			// Write the model (aka the Classifier for further use)
			serializer.writeValueOnFile(f,
					"src/main/resources/qc/classificationAlgorithm.klp");

			// classify examples and compute some statistics
			MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(classes);
			for (Example e : testSet.getExamples()) {
				ClassificationOutput p = f.predict(testSet.getNextExample());
				evaluator.addCount(e, p);
			}
			evaluator.compute();
			
			System.out
					.println("Accuracy: "
							+ evaluator.getAccuracy());
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}
}
