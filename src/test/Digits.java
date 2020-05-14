package test;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import javax.imageio.ImageIO;
import neuralnetwork.Model;
import neuralnetwork.model.ConvLayerDescriptor;
import neuralnetwork.model.DenseLayerDescriptor;
import neuralnetwork.model.LayerDescriptor;
import neuralnetwork.trainer.Trainer;

public class Digits {
	
	private static double[][][] loadData(BufferedImage bm) throws FileNotFoundException {
		int sqSz = 10;
		int nrows = 6;
		double[][] in = new double[10*nrows][sqSz*sqSz];
		double[][] out = new double[10*nrows][10];
		for(int i = 0; i < 10*nrows; i++) {
			for(int j = 0; j < 10; j++) {
				out[i][j] = (i%10) == j ? 1 : 0;
			}
		}
		for(int ny = 0; ny < nrows; ny++) {
			for(int nx = 0; nx < 10; nx++) {
				for(int x = 0; x < sqSz; x++) {
					for(int y = 0; y < sqSz; y++) {
						in[nx+10*ny][y*sqSz+x] = (bm.getRGB(nx*(sqSz+1) + x, ny*(sqSz+1) + y) % 256)/256.0;
					}
				}
			}
		}
		return new double[][][] {in, out};
	}
	
	public static void main(String[] args) throws Exception {
		File folder = new File("C:/Users/Valde/Desktop/digits/"); // folder with training data
		File modelFile = new File(folder, "m2.ai");
		Model model;
		Trainer t;
		if(modelFile.exists()) {
			ObjectInputStream oin = new ObjectInputStream(new FileInputStream(modelFile));
			model = (Model) oin.readObject();
			oin.close();
			t = new Trainer(model);
		}
		else {
			ConvLayerDescriptor dc1 = new ConvLayerDescriptor(10,10,1,3,1,4);
			ConvLayerDescriptor dc2 = new ConvLayerDescriptor(dc1.outW,dc1.outH,dc1.depth,8,1,10);
			model = new Model(new LayerDescriptor[] {
					dc1,
					dc2,
					new DenseLayerDescriptor(dc2.getOutputCount(), 10)
			});
			t = new Trainer(model);
			t.randomizeWeights(0.1);
		}
		double[][][] data = loadData(ImageIO.read(new File(folder, "digits.png")));
		
		KernelDisplay disp1 = new KernelDisplay((ConvLayerDescriptor)model.getLayerDescriptor(0), model.params[0]);
		KernelDisplay disp2 = new KernelDisplay((ConvLayerDescriptor)model.getLayerDescriptor(1), model.params[1]);
		//*
		long time0 = 0;
		int imax = 10000;
		for(int i = 0; i < imax; i+=200) {
			t.trainGDB(200, 0.5, data[0], data[1]);
			long time1 = System.currentTimeMillis();
			if(time1 - time0 > 500) {
				System.out.println("Progress: " + (int)Math.round(100*i/(double)imax) + "%" + " (iteration " + i + " of " + imax + ")");
				System.out.println("Cost:     " + t.cost(data[0], data[1]));
				modelFile.createNewFile();
				ObjectOutputStream oout = new ObjectOutputStream(new FileOutputStream(modelFile));
				oout.writeObject(model);
				oout.close();
				time0 = System.currentTimeMillis();
				disp1.repaint();
				disp2.repaint();
			}
		}
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				System.out.print(Math.round(100*model.params[0][j+3*i])/100.0 + "   ");
			}
			System.out.println();
		}
		/*/
		for(int i = 0; i < 10; i++) {
			System.out.println(i + ": " + Arrays.toString(t.getOutput(data[0][i])));
		}
		//*/
	}
}
