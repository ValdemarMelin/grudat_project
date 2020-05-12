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
	@SuppressWarnings("unused")
	private final Model model;
	private final double[][] params, buffer;
	private final double[][] id;
	
	/**
	 * The last calculated value of the gradient.
	 */
	public final double[][] grad;
	
	/**
	 * Create a trainer from a neural network model.
	 * @param model - the model.
	 */
	public Trainer(Model model) {
		this.model = model;
		this.params = model.params;
		this.buffer = new double[params.length][];
		grad = new double[model.getLayerCount()][];
		layers = new Layer[model.getLayerCount()];
		nwOutC = model.getLayerDescriptor(model.getLayerCount()-1).getOutputCount();
		for(int l = 0; l < model.getLayerCount(); l++)
		{
			buffer[l] = new double[params[l].length];
			grad[l] = new double[params[l].length];
			if(model.getLayerDescriptor(l) instanceof DenseLayerDescriptor) {
				layers[l] = new DenseLayer((DenseLayerDescriptor) model.getLayerDescriptor(l), params[l], nwOutC);
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
		for(int l = 0; l < params.length; l++) {
			for(int w = 0; w < params[l].length; w++) {
				params[l][w] = Math.random()*2*iv - iv;
			}
		}
	}
	
	/**
	 * Performs a specified number of iterations of gradient descend with backtracking. Does NOT update the model.
	 * Note that this method can be implemented using only the public interface of this class. 
	 * @param iterCount the number of iterations.
	 * @param beta the value of the backtracking parameter.
	 */
	public void trainGDB(int iterCount, double beta, double[][] inputs, double[][] outputs) {
		if(beta <= 0 || beta >= 1) throw new IllegalArgumentException("Argument beta must be in range (0,1)");
		double lastCost = cost(inputs, outputs);
		double newCost;
		double t;
		for(int i = 0; i < iterCount; i++) {
			eval(inputs, outputs);
			mark();
			t = 1;
			step(t);
			while((newCost = cost(inputs, outputs)) > lastCost - t/2*Math.pow(gradNorm(),2)) {
				back();
				t *= beta;
				step(t);
			}
			lastCost = newCost;
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
				params[l][w] -= s*grad[l][w];
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
	
	/**
	 * 
	 * @return the norm of the last calculated gradient.
	 */
	public double gradNorm() {
		double g2 = 0;
		for(int l = 0; l < grad.length; l++) {
			for(int w = 0; w < grad[l].length; w++) {
				g2 += grad[l][w]*grad[l][w];
			}
		}
		return Math.sqrt(g2);
	}

	/**
	 * Saves the current weight values in a separate buffer.
	 */
	public void mark() {
		for(int i = 0; i < buffer.length; i++) {
			for(int j = 0; j < buffer[i].length; j++) {
				buffer[i][j] = params[i][j];
			}
		}
	}
	
	/**
	 * Returns to the saved weight values in the buffer.
	 */
	public void back() {
		for(int i = 0; i < buffer.length; i++) {
			for(int j = 0; j < buffer[i].length; j++) {
				params[i][j] = buffer[i][j];
			}
		}
	}
}
