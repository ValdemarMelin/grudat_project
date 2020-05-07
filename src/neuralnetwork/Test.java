package neuralnetwork;


public class Test {
	
	/**
	 * Example: trains a logical and-gate with 3 neurons.
	 * @param args
	 */
	public static void main(String[] args) {
		double[][] inputs = new double[][] {
			new double[] {1,0},
			new double[] {0,1},
			new double[] {1,1},
			new double[] {0,0}
		};
		double[][] outputs = new double[4][1];
		for(int i = 0; i < outputs.length; i++) {
			outputs[i][0] = ((int)Math.round(inputs[i][0] + inputs[i][1])) == 2 ? 1 : 0;
		}
		Trainer t = new Trainer(new int[] {2, 2, 1});
		t.randomWeights(0.1);
		for(int i = 0; i < 10000; i++) {
			t.evaluate(inputs, outputs);
			t.step(1);
		}
		double[] out = new double[1];
		for(int i = 0; i < inputs.length; i++) {
			t.getOutput(inputs[i], out);
			System.out.println(inputs[i][0] + " and " + inputs[i][1] + " = " + out[0]);
		}
	}
}
