/*
 * Copyright 2014 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.uniroma2.sag.kelp.examples.main;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

import java.util.ArrayList;
import java.util.List;

/**
 * This example illustrates how to perform multiclass classification, with a
 * One-Vs-All strategy with SVM.
 * 
 * @author Giuseppe Castellucci, Danilo Croce
 */
public class OneVsAllSVMExample {

	public static void main(String[] args) {
		try {
			// Read a dataset into a trainingSet variable
			SimpleDataset trainingSet = new SimpleDataset();
			trainingSet
					.populate("src/main/resources/iris_dataset/iris_train.klp");

			SimpleDataset testSet = new SimpleDataset();
			testSet.populate("src/main/resources/iris_dataset/iris_test.klp");

			// print some statistics
			System.out.println("Training set statistics");
			System.out.print("Examples number ");
			System.out.println(trainingSet.getNumberOfExamples());

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

			// Kernel for the first representation (0-index)
			Kernel linear = new LinearKernel("0");
			// Normalize the linear kernel
			NormalizationKernel normalizedKernel = new NormalizationKernel(
					linear);
			// instantiate an svmsolver
			BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
			svmSolver.setKernel(normalizedKernel);
			svmSolver.setCp(1);
			svmSolver.setCn(1);

			OneVsAllLearning ovaLearner = new OneVsAllLearning();
			ovaLearner.setBaseAlgorithm(svmSolver);
			ovaLearner.setLabels(classes);

			// learn and get the prediction function
			ovaLearner.learn(trainingSet);
			Classifier f = ovaLearner.getPredictionFunction();

			// classify examples and compute some statistics
			MulticlassClassificationEvaluator ev = new MulticlassClassificationEvaluator(
					trainingSet.getClassificationLabels());

			for (Example e : testSet.getExamples()) {
				ClassificationOutput p = f.predict(testSet.getNextExample());
				ev.addCount(e, p);
			}

			List<Label> twoLabels = new ArrayList<Label>();
			twoLabels.add(new StringLabel("iris-setosa"));
			twoLabels.add(new StringLabel("iris-virginica"));

			Object[] as = new Object[1];
			as[0] = twoLabels;

			System.out.println("Mean F1: "
					+ ev.getPerformanceMeasure("MeanF1"));
			System.out.println("Mean F1 For iris-setosa/iris-virginica: "
					+ ev.getPerformanceMeasure("MeanF1For", as));
			System.out.println("F1: "
					+ ev.getPerformanceMeasure("OverallF1"));
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}
}
