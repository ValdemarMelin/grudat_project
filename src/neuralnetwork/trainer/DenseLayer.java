package neuralnetwork.trainer;

import neuralnetwork.MathUtilities;
import neuralnetwork.model.DenseLayerDescriptor;

final class DenseLayer implements Layer {
	
	private final int inputCount, outputCount;
	private final double[] weights;
	private final double[] stimuli;
	private final double[] output;
	private final double[][] jacobian;
	private final double[][] chainedJacobian;
	private final double[][] chainedWeightJacobian;
	private double[] input;

	public DenseLayer(DenseLayerDescriptor layerDescriptor, double[] weights, int nwOutC) {
		this.inputCount = layerDescriptor.getInputCount();
		this.outputCount = layerDescriptor.getOutputCount();
		this.weights = weights;
		this.stimuli = new double[layerDescriptor.getOutputCount()];
		this.output = new double[layerDescriptor.getOutputCount()];
		this.jacobian = new double[layerDescriptor.getOutputCount()][layerDescriptor.getInputCount()];
		this.chainedJacobian = new double[nwOutC][layerDescriptor.getInputCount()];
		this.chainedWeightJacobian = new double[nwOutC][layerDescriptor.getWeightCount()];
		this.input = null;
	}

	@Override
	public double[][] getWeightDerivative() {
		return chainedWeightJacobian;
	}

	@Override
	public double[] fprop(double[] input) {
		this.input = input;
		for(int n = 0; n < outputCount; n++) {
			stimuli[n] = weights[n*(inputCount+1) + inputCount];
			for(int w = 0; w < inputCount; w++) {
				stimuli[n] += weights[n*(inputCount+1) + w] * input[w];
			}
			output[n] = MathUtilities.f(stimuli[n]);
			final double fprime = MathUtilities.dfdx(stimuli[n]);
			for(int w = 0; w < inputCount; w++) {
				jacobian[n][w] = weights[n*(inputCount+1) + w] * fprime;
			}
		}
		return output;
	}

	@Override
	public double[][] bprop(double[][] chain) {
		MathUtilities.matrixProduct(chain, jacobian, chainedJacobian);
		for(int o = 0; o < chainedWeightJacobian.length; o++) {
			for(int wi = 0; wi < chainedWeightJacobian[o].length; wi++) {
				chainedWeightJacobian[o][wi] = 0;
			}
		}
		for(int o = 0; o < chainedWeightJacobian.length; o++) {
			for(int n = 0; n < outputCount; n++) {
				final double fprime = MathUtilities.dfdx(stimuli[n]);
				for(int w = 0; w < inputCount; w++) {
					chainedWeightJacobian[o][n*(inputCount+1) + w] += chain[o][n]*fprime*input[w];
				}
				chainedWeightJacobian[o][n*(inputCount+1) + inputCount] = fprime*chain[o][n];
			}
		}
		return chainedJacobian;
	}


}
