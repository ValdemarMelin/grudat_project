package test.digits;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

import neuralnetwork.Model;
import neuralnetwork.model.ConvLayerDescriptor;
import neuralnetwork.model.DenseLayerDescriptor;
import neuralnetwork.model.LayerDescriptor;
import neuralnetwork.trainer.Trainer;
import neuralnetwork.utils.KernelDisplay;

public class Digits {
	
	private static double[][][] loadData(BufferedImage bm) throws FileNotFoundException {
		int sqSz = 15;
		int nrows = 1;
		double[][] in = new double[nrows*10*25][sqSz*sqSz];
		double[][] out = new double[nrows*10*25][10];
		for(int digit = 0; digit < 10; digit++) {
			for(int row = 0; row < nrows; row++) {
				for(int sx = 0; sx < 5; sx++) {
					for(int sy = 0; sy < 5; sy++) {
						out[digit * 25*nrows + row*25 + sy*5 + sx][digit] = 1;
						Arrays.fill(in[digit * 25*nrows + row*25 + sy*5 + sx], 1);
					}
				}
				for(int x = 0; x < 10; x++) {
					for(int y = 0; y < 10; y++) {
						double k = (bm.getRGB(x+digit*11, y+row*11) & 0xFF)/255.0;
						for(int sx = 0; sx < 5; sx++) {
							for(int sy = 0; sy < 5; sy++) {
								in[digit * 25*nrows + row*25 + sy*5 + sx][x + sx + 15*(y+sy)] = k;
							}
						}
					}
				}
			}
		}
		return new double[][][] {in, out};
	}
	
	public static void main(String[] args) throws Exception {
		File folder = new File("C:/Users/Valde/Desktop/digits/"); // folder with training data
		File modelFile = new File(folder, "mmmm.ai");
		Model model;
		Trainer t;
		if(modelFile.exists()) {
			ObjectInputStream oin = new ObjectInputStream(new FileInputStream(modelFile));
			model = (Model) oin.readObject();
			oin.close();
			t = new Trainer(model);
		}
		else {
			ConvLayerDescriptor dc1 = new ConvLayerDescriptor(15,15,1,8,1,10);
			model = new Model(new LayerDescriptor[] {
					dc1,
					new DenseLayerDescriptor(dc1.getOutputCount(), 10)
			});
			t = new Trainer(model);
			t.randomizeWeights(0.1);
		}
		double[][][] data = loadData(ImageIO.read(new File(folder, "digits.png")));
		JFrame dispWnd = KernelDisplay.window(model);
		
		long time0 = 0;
		int imax = 10000;
		for(int i = 0; i < imax; i+=10) {
			t.trainGDB(10, 0.5, data[0], data[1]);
			long time1 = System.currentTimeMillis();
			if(time1 - time0 > 500) {
				System.out.println("Progress: " + (int)Math.round(100*i/(double)imax) + "%" + " (iteration " + i + " of " + imax + ")");
				System.out.println("Cost:     " + t.cost(data[0], data[1]));
				modelFile.createNewFile();
				ObjectOutputStream oout = new ObjectOutputStream(new FileOutputStream(modelFile));
				oout.writeObject(model);
				oout.close();
				time0 = System.currentTimeMillis();
				dispWnd.repaint();
			}
		}
	}
}
