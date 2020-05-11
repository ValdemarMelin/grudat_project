package test;

import neuralnetwork.Model;
import neuralnetwork.model.DenseLayerDescriptor;
import neuralnetwork.model.LayerDescriptor;
import neuralnetwork.trainer.Trainer;

/**
 * Trains a simple neural network with structure [2 - 1] ie 1 neuron with 2 inputs and 1 output.
 * @author Valdemar Melin
 */
public class TrainerTest {
	public static void main(String[] args) {
		Model model = new Model(new LayerDescriptor[] {
				new DenseLayerDescriptor(2, 1)
		});
		Trainer trainer = new Trainer(model);
		trainer.randomizeWeights(0.1);
		double[][] inputs = new double[][] {
			new double[] {1, 1},
			new double[] {0, 1},
			new double[] {1, 0},
			new double[] {0, 0},
		};
		double[][] outputs = new double[][] {
			new double[] {1},
			new double[] {0},
			new double[] {0},
			new double[] {0},
		};
		for(int i = 0; i < 1000000; i++) {
			trainer.eval(inputs, outputs);
			trainer.step(0.1);
			if(i % 100000 == 0) {
				
				System.out.println(trainer.cost(inputs, outputs));
			}
		}
		for(double[] input : inputs) {
			System.out.println(input[0] + " & " + input[1] + " = " + trainer.getOutput(input)[0]);
		}
	}
}
