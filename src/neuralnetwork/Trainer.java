package neuralnetwork;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * A network trainer class. 
 * Encapsulates the neural network and forward/backward propagation of it.
 * @author Valdemar Melin
 *
 */
public class Trainer {
	public double[][][] weights;
	public double[][] biases;
	
	double[][] stimuli;
	double[][] output;
	
	double[][][] singleLayerJacobian;
	double[][][] outputLayerJacobian;
	
	double[][][] gradient; // Stores the components of the gradient with respect to each output.
	
	/**
	 * Create a new trainer for neural network.
	 * @param networkStructure: describes the number of inputs(index 0) followed by
	 * the number of neurons of each layer.
	 */
	public Trainer(int[] networkStructure) {
		if(networkStructure == null) return;
		init(networkStructure);
	}
	
	/**
	 * Create an empty trainer
	 */
	public Trainer() {
		
	}
	

	/**
	 * Create an empty trainer
	 * @throws IOException 
	 */
	public Trainer(InputStream in) throws IOException {
		read(in);
	}
	
	/**
	 * Initializes the trainer for a new neural network model.
	 * @param networkStructure: describes the number of inputs(index 0) followed by
	 * the number of neurons of each layer.
	 */
	public void init(int[] networkStructure) {
		weights = new double[networkStructure.length-1][][];
		biases = new double[networkStructure.length-1][];
		stimuli = new double[networkStructure.length-1][];
		output = new double[networkStructure.length-1][];
		singleLayerJacobian = new double[networkStructure.length-1][][];
		outputLayerJacobian = new double[networkStructure.length-1][][];
		gradient = new double[networkStructure.length-1][][];
		for(int l = 0; l < networkStructure.length-1; l++) {
			int neuronCount = networkStructure[l+1], inputCount = networkStructure[l];
			weights[l] = new double[neuronCount][inputCount];
			biases[l] = new double[neuronCount];
			stimuli[l] = new double[neuronCount];
			output[l] = new double[neuronCount];
			singleLayerJacobian[l] = new double[neuronCount][inputCount];
			outputLayerJacobian[l] = new double[networkStructure[networkStructure.length-1]][neuronCount];
			gradient[l] = new double[neuronCount][inputCount+1];
		}
		MathUtilities.matrixIdentity(outputLayerJacobian[outputLayerJacobian.length-1]);
	}
	
	/**
	 * Loads a training model with weights from an input stream.
	 * @param s - the stream
	 * @throws IOException 
	 */
	public void read(InputStream s) throws IOException {
		DataInputStream is = new DataInputStream(s);
		int lcount = is.readInt();
		int[] lsz = new int[lcount+1];
		for(int i = 0; i < lsz.length; i++) {
			lsz[i] = is.readInt();
		}
		init(lsz);
		for(int l = 0; l < weights.length; l++) {
			for(int n = 0; n < weights[l].length; n++) {
				for(int w = 0; w < weights[l][n].length; w++) {
					weights[l][n][w] = is.readDouble();
				}
				biases[l][n] = is.readDouble();
			}
		}
	}
	
	/**
	 * Writes a training model with weights to an output stream.
	 * @param s
	 * @throws IOException 
	 */
	public void write(OutputStream s) throws IOException {
		DataOutputStream os = new DataOutputStream(s);
		os.writeInt(weights.length);// layer count
		os.writeInt(weights[0][0].length);//input size
		for(int l = 0; l < weights.length; l++) {
			os.writeInt(weights[l].length);
		}
		for(int l = 0; l < weights.length; l++) {
			for(int n = 0; n < weights[l].length; n++) {
				for(int w = 0; w < weights[l][n].length; w++) {
					os.writeDouble(weights[l][n][w]);
				}
				os.writeDouble(biases[l][n]);
			}
		}
	}
	
	/**
	 * Makes all weights random numbers between -range and range.
	 * @param range
	 */
	public void randomWeights(double range) {
		for(int l = 0; l < weights.length; l++) {
			for(int n = 0; n < weights[l].length; n++) {
				for(int w = 0; w < weights[l][n].length; w++) {
					weights[l][n][w] = Math.random()*2*range - range;
				}
				biases[l][n] = Math.random()*2*range - range;
			}
		}
	}
	
	/**
	 * Evaluates the gradient of the cost function with the specified training data.
	 * @param inputs - the input training data
	 * @param outputs - the output training data 
	 * 
	 * Time complexity: O(M*N*O) where M: number of training data, N: number of weights, O: number of ouputs.
	 */
	public void evaluate(double[][] inputs, double[][] outputs) {
		for(int i = 0; i < gradient.length; i++) {
			for(int j = 0; j < gradient[i].length; j++) {
				for(int k = 0; k < gradient[i][j].length; k++) {
					gradient[i][j][k] = 0;
				}
			}
		}
		
		final int lcount = weights.length;
		final int ocount = weights[weights.length-1].length;
		final double[] r = new double[ocount];
		for(int ii = 0; ii < inputs.length; ii++)
		{
			// forward propagation
			double[] layerInput = inputs[ii];
			for(int li = 0; li < lcount; li++)
			{
				int ncount = weights[li].length;
				for(int ni = 0; ni < ncount; ni++)
				{
					stimuli[li][ni] = biases[li][ni];
					for(int wi = 0; wi < weights[li][ni].length; wi++)
					{
						stimuli[li][ni] += weights[li][ni][wi] * layerInput[wi];
					}
					output[li][ni] = MathUtilities.f(stimuli[li][ni]);
					final double fprime = MathUtilities.dfdx(stimuli[li][ni]);
					for(int wi = 0; wi < weights[li][ni].length; wi++)
					{
						singleLayerJacobian[li][ni][wi] += weights[li][ni][wi] * fprime;
					}
				}
				layerInput = output[li];
			}
			
			for(int oi = 0; oi < ocount; oi++) {
				r[oi] = output[lcount-1][oi] - outputs[ii][oi];
			}
			
			// backward propagation: compute "multi-layer" jacobian.
			for(int li = lcount-2; li >= 0; li--)
			{
				MathUtilities.matrixProduct(outputLayerJacobian[li+1], singleLayerJacobian[li+1], outputLayerJacobian[li]);
			}
			
			// add contribution to gradient
			layerInput = inputs[ii];
			for(int li = 0; li < gradient.length; li++) {
				for(int ni = 0; ni < gradient[li].length; ni++) {
					final double fprime = MathUtilities.dfdx(stimuli[li][ni]);
					for(int k = 0; k < weights[li][ni].length; k++) {
						for(int oi = 0; oi < ocount; oi++)
						{
							gradient[li][ni][k] += outputLayerJacobian[li][oi][ni]*layerInput[k]*fprime*r[oi];
						}
					}
					// bias
					for(int oi = 0; oi < ocount; oi++)
					{
						gradient[li][ni][weights[li][ni].length] += outputLayerJacobian[li][oi][ni]*fprime*r[oi];
					}
				}
				layerInput = output[li];
			}
		}
	}
	
	/**
	 * @return the norm of the gradient.
	 */
	public double gradientSize() {
		double gnorm = 0;
		for(int i = 0; i < gradient.length; i++) {
			for(int j = 0; j < gradient[i].length; j++) {
				for(int k = 0; k < gradient[i][j].length; k++) {
					gnorm += gradient[i][j][k]*gradient[i][j][k];
				}
			}
		}
		return  Math.sqrt(gnorm);
	}
	
	/**
	 * Computes the value of the cost function.
	 * The cost function is defined as the sum of the difference of the output and the correct output squared.
	 * @param inputs
	 * @param outputs
	 */
	public double cost(double[][] inputs, double[][] outputs) {
		final int lcount = weights.length;
		final int ocount = weights[weights.length-1].length;
		double cost = 0;
		for(int ii = 0; ii < inputs.length; ii++)
		{
			double[] layerInput = inputs[ii];
			for(int li = 0; li < lcount; li++)
			{
				int ncount = weights[li].length;
				for(int ni = 0; ni < ncount; ni++)
				{
					stimuli[li][ni] = biases[li][ni];
					for(int wi = 0; wi < weights[li][ni].length; wi++)
					{
						stimuli[li][ni] += weights[li][ni][wi] * layerInput[wi];
					}
					output[li][ni] = MathUtilities.f(stimuli[li][ni]);
				}
				layerInput = output[li];
			}
			
			for(int oi = 0; oi < ocount; oi++) {
				cost += Math.pow(output[lcount-1][oi] - outputs[ii][oi],2);
			}
		}
		return cost;
	}
	
	/**
	 * Inputs some data to the network and calculates the output.
	 * @param input
	 * @param output
	 */
	public void getOutput(double[] input, double[] output) {
		final int lcount = weights.length;
		final int ocount = weights[weights.length-1].length;
		double[] layerInput = input;
		for(int li = 0; li < lcount; li++)
		{
			int ncount = weights[li].length;
			for(int ni = 0; ni < ncount; ni++)
			{
				stimuli[li][ni] = biases[li][ni];
				for(int wi = 0; wi < weights[li][ni].length; wi++)
				{
					stimuli[li][ni] += weights[li][ni][wi] * layerInput[wi];
				}
				this.output[li][ni] = MathUtilities.f(stimuli[li][ni]);
			}
			layerInput = this.output[li];
		}
		for(int i = 0; i < ocount; i++) {
			output[i] = layerInput[i];
		}
	}
	
	/**
	 * Updates all weights and biases in the direction of -gradient
	 * @param s - the step size
	 */
	public void step(double s) {
		for(int i = 0; i < gradient.length; i++) {
			for(int j = 0; j < gradient[i].length; j++) {
				for(int k = 0; k < gradient[i][j].length-1; k++) {
					weights[i][j][k] -= s*gradient[i][j][k];
				}
				biases[i][j] -= s*gradient[i][j][weights[i][j].length];
			}
		}
	}
}
