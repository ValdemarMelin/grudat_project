package neuralnetwork;

/**
 * Some helper methods for math.
 * @author Valdemar Melin
 *
 */
public class MathUtilities {
	
	/**
	 * Activation function for the neural network: sigmoid function.
	 * @param x
	 * @return
	 */
	public static double f(double x) {
		return 1/(1 + Math.exp(-x));
	}
	
	/**
	 * Derivative of the activation function.
	 * @param x
	 * @return
	 */
	public static double dfdx(double x) {
		double y = f(x);
		return y*(1-y);
	}
	
	/**
	 * Makes an identity matrix.
	 * @param a
	 */
	public static void matrixIdentity(double[][] a) {
		for(int i = 0; i < a.length; i++) {
			for(int j = 0; j < a[i].length; j++) {
				a[i][j] = i == j ? 1 : 0;
			}
		}
	}
	
	/**
	 * Multiplies two matrices and stores the answer in the third.
	 * @param a - one of the matrices to multiply.
	 * @param b - the other.
	 * @param o - the result.
	 */
	public static void matrixProduct(double[][] a, double[][] b, double[][] o) {
		for(int r = 0; r < a.length; r++) {
			for(int c = 0; c < b[0].length; c++) {
				o[r][c] = 0;
				for(int k = 0; k < a[0].length; k++) {
					o[r][c] += a[r][k] * b[k][c];
				}
			}
		}
	}
}
