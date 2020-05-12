package test;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;

import javax.imageio.ImageIO;

import neuralnetwork.Model;
import neuralnetwork.trainer.Trainer;

/**
 * Tests the network trained in Digits.java file. Place the network and a 5x5 pixel gray-scale digit image
 * in a directory and name them "model5.ai" and "test.png" and run the code ( you need to enter the directory in the source).
 * @author valde
 *
 */
public class TestDigits {
	public static void main(String[] args) throws Exception {
		File dir = new File("C:/"); // insert directory to network and test image here
		ObjectInputStream oin = new ObjectInputStream(new FileInputStream(new File(dir,"model5.ai")));
		Model model = (Model) oin.readObject();
		oin.close();
		Trainer t = new Trainer(model);
		BufferedImage inputImage = ImageIO.read(new File(dir, "test.png"));
		double[] input = new double[25];
		for(int i = 0; i < 5; i++) {
			for(int j = 0; j < 5; j++) {
				input[i*5 + j] = (inputImage.getRGB(j, i) % 256)/256.0;
			}
		}
		double[] o = t.getOutput(input);
		int imax = -1; double max = 0;
		for(int i = 0; i < 10; i++) {
			if(o[i] > max) {
				max = o[i];
				imax = i;
			}
		}
		System.out.println((imax + 1)%10);
	}
}