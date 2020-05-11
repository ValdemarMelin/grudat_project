package neuralnetwork.model;

/**
 * Layer descriptor for a dense (fully connected) layer.
 * @author Valdemar Melin
 *
 */
public class DenseLayerDescriptor implements LayerDescriptor {
	private static final long serialVersionUID = -8755645900557022701L;
	
	public final int inputCount, outputCount;
	
	/**
	 * Create a new dense layer descriptor with specified number of inputs and outputs.
	 * @param inputCount
	 * @param outputCount
	 */
	public DenseLayerDescriptor(int inputCount, int outputCount) {
		this.inputCount = inputCount;
		this.outputCount = outputCount;
	}

	@Override
	public int getInputCount() {
		return inputCount;
	}

	@Override
	public int getOutputCount() {
		return outputCount;
	}

	@Override
	public int getWeightCount() {
		return (inputCount+1)*outputCount;
	}
	
	@Override
	public boolean equals(Object o) {
		if(o instanceof DenseLayerDescriptor) {
			DenseLayerDescriptor d2 = (DenseLayerDescriptor) o;
			return d2.inputCount == inputCount && d2.outputCount == outputCount;
		}
		else return false;
	}
}
