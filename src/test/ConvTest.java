package test;


import java.util.Arrays;

import neuralnetwork.Model;
import neuralnetwork.model.ConvLayerDescriptor;
import neuralnetwork.model.LayerDescriptor;
import neuralnetwork.trainer.ConvLayer;
import neuralnetwork.trainer.Trainer;

public class ConvTest {
	
	public static void main(String[] args) throws Exception {
		ConvLayerDescriptor cd = new ConvLayerDescriptor(10, 10, 1, 3, 1, 1);
		
		Model m = new Model(new LayerDescriptor[] {cd});
		ConvLayer cl = new ConvLayer(cd, new double[] {-1,-1,-1,-1,8,-1,-1,-1,-1}, 8*8*1);
		double[][] in = new double[10][10*10];
		double[][] out = new double[10][8*8];
		for(int i = 0; i < 10; i++) {
			for(int j = 0; j < 10*10; j++) {
				in[i][j] = Math.random();
			}
		}
		for(int i = 0; i < 10; i++) {
			double[] k = cl.fprop(in[i]);
			for(int j = 0; j < k.length; j++) {
				out[i][j] = k[j];
			}
		}
		Trainer t = new Trainer(m);
		t.randomizeWeights(0.1);
		System.out.println(Arrays.toString(m.params[0]));
		for(int i = 0; i < 10; i++) {
			long t0 = System.currentTimeMillis();
			t.trainGDB(100, 0.2, in, out);
			long t1 = System.currentTimeMillis();
			System.out.println(t1 - t0);
		}
		System.out.println(Arrays.toString(m.params[0]));
	}
}
