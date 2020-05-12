package neuralnetwork;

import java.io.Serializable;
import java.util.Arrays;

import neuralnetwork.model.LayerDescriptor;

/**
 * A class containing all information about a neural network and its weights.
 * Can be serialized to save/share the model and weights. Can be trained by creating
 * a Trainer object.
 * @author Valdemar Melin
 * @see neuralnetwork.trainer.Trainer
 */
public class Model implements Serializable {
	private static final long serialVersionUID = -399505443432890425L;
	
	private final LayerDescriptor[] modelDescriptor;
	
	/**
	 * The current weights/parameters of the model.
	 */
	public final double[][] params;
	
	public Model(LayerDescriptor[] modelDescriptor) {
		this.modelDescriptor = modelDescriptor.clone();
		this.params = new double[modelDescriptor.length][];
		for(int l = 0; l < modelDescriptor.length; l++) {
			this.params[l] = new double[modelDescriptor[l].getWeightCount()];
		}
	}
	
	/**
	 * 
	 * @param layer - the index of the layer in the model.
	 * @return the LayerDescriptor for the layer.
	 */
	public LayerDescriptor getLayerDescriptor(int layer) {
		return modelDescriptor[layer];
	}
	
	/**
	 * 
	 * @return the number of weights in the model.
	 */
	public final int getLayerCount() {
		return params.length;
	}
	
	@Override
	public boolean equals(Object o) {
		if(o instanceof Model) {
			Model m2 = (Model)o;
			if(m2.modelDescriptor.length != modelDescriptor.length)
				return false;
			for(int l = 0; l < modelDescriptor.length; l++) {
				if(!Arrays.equals(m2.params[l], params[l]) || !modelDescriptor[l].equals(m2.modelDescriptor[l]))
					return false;
			}
			return true;
		}
		else return false;
	}
}
