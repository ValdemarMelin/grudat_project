package test;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.Scanner;

import javax.imageio.ImageIO;

import neuralnetwork.Model;
import neuralnetwork.model.DenseLayerDescriptor;
import neuralnetwork.model.LayerDescriptor;
import neuralnetwork.trainer.Trainer;

public class Digits {
	
	private static double[][][] loadData(BufferedImage bm) throws FileNotFoundException {
		double[][] in = new double[10][25];
		double[][] out = new double[10][10];
		for(int i = 0; i < 10; i++) {
			for(int j = 0; j < 10; j++) {
				out[i][j] = i == j ? 1 : 0;
			}
		}
		for(int n = 0; n < 10; n++) {
			for(int i = 0; i < 5; i++) {
				for(int j = 0; j < 5; j++) {
					in[n][i*5+j] = (bm.getRGB(n*6 + j, i) % 256)/256.0;
				}
			}
		}
		return new double[][][] {in, out};
	}
	
	public static void main(String[] args) throws Exception {
		File folder = new File("C:/Users/Valde/Desktop/digits/");
		File modelFile = new File(folder, "model3.ai");
		Model model;
		if(modelFile.exists()) {
			ObjectInputStream oin = new ObjectInputStream(new FileInputStream(modelFile));
			model = (Model) oin.readObject();
			oin.close();
		}
		else {
			model = new Model(new LayerDescriptor[] {
					new DenseLayerDescriptor(5*5,12),
					new DenseLayerDescriptor(12, 10)
			});
		}
		double[][][] data = loadData(ImageIO.read(new File(folder, "digits.png")));
		Trainer t = new Trainer(model);
		long time0 = 0;
		int imax = 1000000;
		for(int i = 0; i < imax; i++) {
			t.eval(data[0], data[1]);
			t.step(0.01);
			long time1 = System.currentTimeMillis();
			if(time1 - time0 > 5000) {
				System.out.println("Progress: " + (int)Math.round(100*i/(double)imax) + "%" + " (iteration " + i + " of " + imax + ")");
				System.out.println("Cost:     " + t.cost(data[0], data[1]));
				modelFile.createNewFile();
				ObjectOutputStream oout = new ObjectOutputStream(new FileOutputStream(modelFile));
				oout.writeObject(model);
				oout.close();
				time0 = System.currentTimeMillis();
			}
		}
		/*
		for(int i = 0; i < 10; i++) {
			System.out.println(i + ": " + Arrays.toString(t.getOutput(data[0][i])));
		}
		*/
	}
}
