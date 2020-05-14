package neuralnetwork.model;

/**
 * Layer descriptor for a convolution layer.
 * @author Valdemar Melin
 *
 */
public class ConvLayerDescriptor implements LayerDescriptor{
	private static final long serialVersionUID = -3188673529618377755L;
	
	public final int inW, inH, inD;
	public final int outW, outH;
	public final int size, stride, depth;

	/**
	 * Create a new convolution layer descriptor.
	 * @param inW - the width of the input data "image"
	 * @param inH - the height of the input data "image"
	 * @param inD - the pixel/channel depth of the input 
	 * @param size - the size of the kernel(side length)
	 * @param stride - the step size the kernel moves with across input data.
	 * @param depth - the number of kernels.
	 */
	public ConvLayerDescriptor(int inW, int inH, int inD, int size, int stride, int depth) {
		this.inW = inW;
		this.inH = inH;
		this.inD = inD;
		this.size = size;
		this.stride = stride;
		this.depth = depth;
		this.outW = (inW - size)/stride + 1;
		this.outH = (inH - size)/stride + 1;
	}

	@Override
	public int getInputCount() {
		return inW*inH*inD;
	}

	@Override
	public int getOutputCount() {
		return outW*outH*depth;
	}

	@Override
	public int getWeightCount() {
		return size*size*inD*depth;
	}
}
