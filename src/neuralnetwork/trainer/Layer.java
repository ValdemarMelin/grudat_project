package neuralnetwork.trainer;

interface Layer {
	double[] fprop(double[] input);
	double[][] bprop(double[][] chain);
	double[][] getWeightDerivative();
}
