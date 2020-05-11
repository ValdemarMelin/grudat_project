package neuralnetwork.trainer;

import neuralnetwork.MathUtilities;
import neuralnetwork.Model;
import neuralnetwork.model.DenseLayerDescriptor;

/**
 * Provides methods for performing gradient descend on a neural network, with
 * methods for computing gradient and update weights accordingly.
 * @see test test package for usage.
 * @author Valdemar Melin
 *
 */
public class Trainer {
	
	private final Layer[] layers;
	private final int nwOutC;
	private final Model model;
	private final double[][] grad;
	private final double[][] id;
	
	/**
	 * Create a trainer from a neural network model.
	 * @param model - the model.
	 */
	public Trainer(Model model) {
		this.model = model;
		grad = new double[model.getLayerCount()][];
		layers = new Layer[model.getLayerCount()];
		nwOutC = model.getLayerDescriptor(model.getLayerCount()-1).getOutputCount();
		for(int l = 0; l < model.getLayerCount(); l++)
		{
			grad[l] = new double[model.getParams(l).length];
			if(model.getLayerDescriptor(l) instanceof DenseLayerDescriptor) {
				layers[l] = new DenseLayer((DenseLayerDescriptor) model.getLayerDescriptor(l), model.getParams(l), nwOutC);
			}
			// TODO: add other layer types
			else {
				throw new RuntimeException("Unsupported layer type");
			}
		}
		id = new double[nwOutC][nwOutC];
		MathUtilities.matrixIdentity(id);
	}
	
	/**
	 * Randomizes the model's weights in the interval [-iv, iv].
	 * @param iv - the largest absolute value of the weights after the randomization.
	 */
	public void randomizeWeights(double iv) {
		for(int l = 0; l < layers.length; l++) {
			for(int w = 0; w < model.getParams(l).length; w++) {
				model.getParams(l)[w] = Math.random()*2*iv - iv;
			}
		}
	}
	
	/**
	 * Computes the gradient of the cost function with respect to the specified training data.
	 * @param inputs - an array containing the input data.
	 * @param outputs - an array containing the output data.
	 */
	public void eval(double[][] inputs, double[][] outputs) {
		// Zero gradient
		for(int l = 0; l < grad.length; l++)
			for(int w = 0; w < grad[l].length; w++) grad[l][w] = 0;
		
		for(int ii = 0; ii < inputs.length; ii++)
		{
			// Forward propagation
			double[] in = inputs[ii];
			for(int l = 0; l < layers.length; l++) {
				in = layers[l].fprop(in);
			}
			
			// Backward propagation
			double[][] chain = id;
			for(int l = layers.length-1; l >= 0; l--) {
				chain = layers[l].bprop(chain);
			}
			
			// Compute gradient
			for(int o = 0; o < nwOutC; o++) {
				final double dd = in[o] - outputs[ii][o];
				for(int l = 0; l < grad.length; l++) {
					for(int w = 0; w < grad[l].length; w++) {
						grad[l][w] += dd*layers[l].getWeightDerivative()[o][w];
					}
				}
			}
		}
	}
	
	/**
	 * 
	 * Computes the cost function of the specified training data.
	 * @param inputs - an array containing the input data.
	 * @param outputs - an array containing the output data.
	 */
	public double cost(double[][] inputs, double[][] outputs) {
		double c = 0;
		for(int ii = 0; ii < inputs.length; ii++)
		{
			// Forward propagation
			double[] in = inputs[ii];
			for(int l = 0; l < layers.length; l++) {
				in = layers[l].fprop(in);
			}
			for(int o = 0; o < nwOutC; o++) {
				c += (in[o] - outputs[ii][o])*(in[o] - outputs[ii][o]);
			}
		}
		return c;
	}
	
	/**
	 * Takes a step in the opposite direction of the gradient with a factor s.
	 * @param s - the step scale factor.
	 */
	public void step(double s) {
		for(int l = 0; l < grad.length; l++) {
			for(int w = 0; w < grad[l].length; w++) {
				model.getParams(l)[w] -= s*grad[l][w];
			}
		}
	}

	/**
	 * Returns the output of the model with current weights
	 * @param input - the input to the model
	 * @return - the output of the model.
	 */
	public double[] getOutput(double[] input) {
		double[] out = input;
		for(int l = 0; l < layers.length; l++) {
			out = layers[l].fprop(out);
		}
		return out;
	}
}
