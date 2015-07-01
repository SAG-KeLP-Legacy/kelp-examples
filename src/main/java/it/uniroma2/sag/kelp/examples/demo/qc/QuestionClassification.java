package it.uniroma2.sag.kelp.examples.demo.qc;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexKernelCache;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexSquaredNormCache;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.tree.PartialTreeKernel;
import it.uniroma2.sag.kelp.kernel.tree.SubSetTreeKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

/**
 * This class shows how to use Kelp to build a question classifer. In the
 * example, three different kernels can be used to build your own classifier.
 * 
 * @author Simone Filice, Giuseppe Castellucci, Danilo Croce
 * 
 */
public class QuestionClassification {

	public static void main(String[] args) {
		try {
			// Initializing the Log level
			System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "WARN");

			// Read both training and testing dataset
			SimpleDataset trainingSet = new SimpleDataset();
			trainingSet
					.populate("src/main/resources/qc/train_5500.coarse.klp.gz");
			SimpleDataset testSet = new SimpleDataset();
			testSet.populate("src/main/resources/qc/TREC_10.coarse.klp.gz");

			/*
			 * ATTENTION
			 * 
			 * Use this parameter to use:
			 * 
			 * - stk: a Subset Tree Kernel over a tree derived from the
			 * dependency parse tree.
			 * 
			 * - bow: a Linear Kernel applied to a boolean Bag-of-Word vector,
			 * where each boolean dimension indicates the presence of the
			 * corresponding word in the question.
			 * 
			 * - ptk: a Subset Tree Kernel over a tree derived from the
			 * dependency parse tree.
			 * 
			 * 
			 * More details on the tree kernel can be found in the corresponding
			 * class. A description of the tree construction, called Grammatical
			 * Relation Centered Tree can be found in:
			 * 
			 * [Croce et al(2011)] Croce D., Moschitti A., Basili R. (2011)
			 * Structured lexical similarity via convolution kernels on
			 * dependency trees. In: Proceedings of EMNLP, Edinburgh, Scotland.
			 */
			String tkString = "bow";

			// print some statistics
			System.out.println("Training set statistics");
			System.out.print("Examples number ");
			System.out.println(trainingSet.getNumberOfExamples());

			// print the number of train and test example for each class
			for (Label l : trainingSet.getClassificationLabels()) {
				System.out.println("Training Label " + l.toString() + " "
						+ trainingSet.getNumberOfPositiveExamples(l));
				System.out.println("Training Label " + l.toString() + " "
						+ trainingSet.getNumberOfNegativeExamples(l));

				System.out.println("Test Label " + l.toString() + " "
						+ testSet.getNumberOfPositiveExamples(l));
				System.out.println("Test Label " + l.toString() + " "
						+ testSet.getNumberOfNegativeExamples(l));
			}

			// Set the cache size
			int cacheSize = trainingSet.getNumberOfExamples()
					+ testSet.getNumberOfExamples();

			// Initialize the kernel function
			Kernel usedKernel = null;
			if (tkString.equalsIgnoreCase("stk")) {
				// Kernel for the first representation (0-index)
				Kernel stkgrct = new SubSetTreeKernel(0.4f, "grct");
				// This cache stores the norm of the kernel used BEFORE the
				// normalization.
				stkgrct.setSquaredNormCache(new FixIndexSquaredNormCache(
						cacheSize));
				// The kernel is normalized.
				Kernel normPtkGrct = new NormalizationKernel(stkgrct);
				usedKernel = normPtkGrct;
			} else if (tkString.equalsIgnoreCase("bow")) {
				Kernel linearKernel = new LinearKernel("bow");
				// This cache stores the norm of the kernel used BEFORE the
				// normalization.
				linearKernel.setSquaredNormCache(new FixIndexSquaredNormCache(
						cacheSize));
				// The kernel is normalized.
				Kernel normLinearKernel = new NormalizationKernel(linearKernel);
				usedKernel = normLinearKernel;
			} else if (tkString.equalsIgnoreCase("ptk")) {
				// Kernel for the first representation (0-index)
				Kernel ptkgrct = new PartialTreeKernel(0.4f, 0.4f, 5f, "grct");
				// This cache stores the norm of the kernel used BEFORE the
				// normalization.
				ptkgrct.setSquaredNormCache(new FixIndexSquaredNormCache(
						cacheSize));
				// The kernel is normalized.
				Kernel normPtkGrct = new NormalizationKernel(ptkgrct);
				usedKernel = normPtkGrct;
			}
			// Set cache to the kernel
			usedKernel.setKernelCache(new FixIndexKernelCache(cacheSize));

			JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();

			// Instantiate the SVM learning Algorithm. This is a binary
			// classifier that will be transparently duplicated from the
			// Multi-class classifier
			BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
			// Set the kernel
			svmSolver.setKernel(usedKernel);
			// Set the C parameter
			svmSolver.setCn(3);
			// Enamble the fairness: in each binary classifier, the
			// regularization parameter of the
			// positive examples is multiplied of a coefficient that is
			// number_of_negative_examples/number_of_positive_examples
			svmSolver.setFairness(true);

			// Instantiate the multi class classifier that apply a One-vs-All
			// schema
			OneVsAllLearning ovaLearner = new OneVsAllLearning();
			// Use the binary classifier defined above
			ovaLearner.setBaseAlgorithm(svmSolver);
			ovaLearner.setLabels(trainingSet.getClassificationLabels());
			// The classifier can be serialized
			serializer.writeValueOnFile(ovaLearner,
					"src/main/resources/qc/learningAlgorithmSpecificationFromJavaCode.klp");

			// Learn and get the prediction function
			ovaLearner.learn(trainingSet);
			Classifier f = ovaLearner.getPredictionFunction();
			// Write the model (aka the Classifier for further use)
			serializer.writeValueOnFile(f,
					"src/main/resources/qc/classificationAlgorithm.klp");

			// Classify examples and compute the accuracy, i.e. the percentage
			// of questions that are correctly classified
			MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(
					trainingSet.getClassificationLabels());
			for (Example e : testSet.getExamples()) {
				// Predict the class
				ClassificationOutput p = f.predict(testSet.getNextExample());
				evaluator.addCount(e, p);
				System.out
						.println("Question:\t" + e.getRepresentation("quest"));
				System.out.println("Original class:\t"
						+ e.getClassificationLabels());
				System.out.println("Predicted class:\t"
						+ p.getPredictedClasses());
				System.out.println();
			}
			evaluator.compute();

			System.out.println("Accuracy: " + evaluator.getAccuracy());
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}
}
