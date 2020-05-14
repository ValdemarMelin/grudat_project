package neuralnetwork.trainer;

import neuralnetwork.MathUtilities;
import neuralnetwork.model.ConvLayerDescriptor;

public class ConvLayer implements Layer {
	private final ConvLayerDescriptor dc;
	private final double[] weights;
	private final double[] input;
	private final double[] output;
	private final double[][] chain;
	private final double[][] wd;

	public ConvLayer(ConvLayerDescriptor dc, double[] weights, int nwOutC) {
		this.dc = dc;
		this.weights = weights;
		this.input = new double[dc.getInputCount()];
		this.output = new double[dc.getOutputCount()];
		this.chain = new double[nwOutC][dc.getInputCount()];
		this.wd = new double[nwOutC][dc.getWeightCount()];
	}
	
	private final int ii(int x, int y, int i) {
		return x + y*dc.inW + i*dc.inW*dc.inH;
	}

	private final int oi(int x, int y, int i) {
		return x/dc.stride + y/dc.stride*dc.outW + i*dc.outW*dc.outH;
	}
	
	private final int wi(int dx, int dy, int id, int od) {
		return dx + dy*dc.size + id*dc.size*dc.size + od*dc.size*dc.size*dc.inD;
	}
	
	@Override
	public double[] fprop(double[] input) {
		for(int i = 0; i < input.length; i++) this.input[i] = input[i];
		for(int d = 0; d < dc.depth; d++) {
			for(int x = 0; x < dc.inW - dc.size + 1; x+=dc.stride) {
				for(int y = 0; y < dc.inH - dc.size + 1; y+=dc.stride) {
					// apply kernel
					output[oi(x,y,d)] = 0;
					for(int c = 0; c < dc.inD; c++) {
						for(int dx = 0; dx < dc.size; dx++) {
							for(int dy = 0; dy < dc.size; dy++) {
								output[oi(x,y,d)] += weights[wi(dx,dy,c,d)]*input[ii(x+dx,y+dy,c)];
							}
						}
					}
				}
			}
		}
		return output;
	}

	@Override
	public double[][] bprop(double[][] chain) {
		MathUtilities.zeroMatrix(this.chain);
		MathUtilities.zeroMatrix(this.wd);
		for(int o = 0; o < chain.length; o++) {
			for(int d = 0; d < dc.depth; d++) {
				for(int x = 0; x < dc.inW - dc.size + 1; x+=dc.stride) {
					for(int y = 0; y < dc.inH - dc.size + 1; y+=dc.stride) {
						// kernel
						for(int c = 0; c < dc.inD; c++) {
							for(int dx = 0; dx < dc.size; dx++) {
								for(int dy = 0; dy < dc.size; dy++) {
									this.chain[o][ii(x+dx,y+dy,c)] += chain[o][oi(x,y,d)]*weights[wi(dx,dy,c,d)];
									this.wd[o][wi(dx,dy,c,d)] += chain[o][oi(x,y,d)]*input[ii(x+dx,y+dy,c)];
								}
							}
						}
					}
				}
			}
		}
		return this.chain;
	}

	@Override
	public double[][] getWeightDerivative() {
		return wd;
	}

}
