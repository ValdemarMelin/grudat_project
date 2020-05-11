package test;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

import neuralnetwork.Model;
import neuralnetwork.model.DenseLayerDescriptor;
import neuralnetwork.model.LayerDescriptor;
import neuralnetwork.trainer.Trainer;

/**
 * Trains a network on a cosine function.
 * The network structure is [1 - 10 - 1], ie 1 input, 10 neurons in first hidden layer
 * and 1 neuron in the output layer - all layers fully connected.
 * 
 * Plots of the result with different numbers of iterations can be found in "tests/TrainerTest2"
 * @author Valdemar Melin
 *
 */
public class TrainerTest2 {
	
	public static double[][] input;
	public static double[][] output;
	
	/*
	 * Plot in Matlab/Octave or other plotting application.
	 */
	private static void saveGraph(Trainer t) throws IOException {
		System.out.println("Please enter a file to save the plot to.");
		Scanner s = new Scanner(System.in);
		File file = new File(s.nextLine());
		s.close();
		file.createNewFile();
		FileWriter fw = new FileWriter(file, false);
		try {
			double[] i = new double[1];
			for(i[0] = -2.5; i[0] < 2.5; i[0]+= 0.05) {
				fw.write(i[0] + " " + t.getOutput(i)[0] + System.lineSeparator());
			}
		}
		finally {
			fw.close();
		}
		return;
	}
	
	private static void makeTrainingData() {
		input = new double[20][1];
		output = new double[20][1];
		for(int i = 0; i < input.length; i++) {
			input[i][0] = (i - 10)/4.0;
			output[i][0] = 0.1 + 0.1*Math.cos((i-10));
		}
	}

	/**
	 * Usage: java TrainerTest2
	 */
	public static void main(String[] args) {
		Model model = new Model(new LayerDescriptor[] {
				new DenseLayerDescriptor(1, 10),
				new DenseLayerDescriptor(10, 1)
		});
		Trainer trainer = new Trainer(model);
		makeTrainingData();
		trainer.randomizeWeights(0.1);
		for(int i = 0; i < 100000; i++) {
			trainer.eval(input, output);
			trainer.step(0.1);
		}
		try {
			saveGraph(trainer);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
