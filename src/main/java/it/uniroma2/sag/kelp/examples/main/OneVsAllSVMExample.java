package it.uniroma2.sag.kelp.examples.main;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;

import java.util.List;

public class OneVsAllSVMExample {

	public static void main(String[] args) {
		try {
			// Read a dataset into a trainingSet variable
			SimpleDataset trainingSet = new SimpleDataset();
			trainingSet.populate("src/main/resources/iris_dataset/iris_train.klp");
			
			SimpleDataset testSet = new SimpleDataset();
			testSet.populate("src/main/resources/iris_dataset/iris_test.klp");

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
			
			// Kernel for the first representation (0-index)
			Kernel linear = new LinearKernel("0");
			// Normalize the linear kernel
			NormalizationKernel normalizedKernel = new NormalizationKernel(
					linear);
			// instantiate an svmsolver
			BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
			svmSolver.setKernel(normalizedKernel);
			svmSolver.setCp(2);
			svmSolver.setCn(1);
			
			OneVsAllLearning ovaLearner = new OneVsAllLearning();
			ovaLearner.setBaseAlgorithm(svmSolver);
			ovaLearner.setLabels(classes);
			
			// learn and get the prediction function
			ovaLearner.learn(trainingSet);
			Classifier f = ovaLearner.getPredictionFunction();

			// classify examples and compute some statistics
			int correct = 0;
			for (Example e : testSet.getExamples()) {
				ClassificationOutput p = f.predict(testSet.getNextExample());
//				System.out.println(p.getPredictedClasses());
				if (e.isExampleOf(p.getPredictedClasses().get(0))) {
					correct++;
				}
			}

			System.out
					.println("Accuracy: "
							+ ((float) correct / (float) testSet
									.getNumberOfExamples()));
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

}
