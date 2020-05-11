package neuralnetwork.model;

import java.io.Serializable;

/**
 * Interface for a layer descriptor, that different types of layers implement.
 * @author Valdemar Melin
 *
 */
public interface LayerDescriptor extends Serializable {
	/**
	 * @return the number of inputs that the layer expects.
	 */
	public int getInputCount();
	
	/**
	 * @return the number of outputs that the layer produces.
	 */
	public int getOutputCount();

	/**
	 * @return the number of independent weights that the layer contains.
	 */
	public int getWeightCount();
}