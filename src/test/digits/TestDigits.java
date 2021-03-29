package test.digits;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.Arrays;

import javax.imageio.ImageIO;

import neuralnetwork.Model;
import neuralnetwork.trainer.Trainer;

/**
 * Tests the network trained in Digits.java file. Place the network and a 5x5 pixel gray-scale digit image
 * in a directory and name them "m3.ai" and "test.png" and run the code ( you need to enter the directory in the source).
 * @author Valdemar Melin
 *
 */
public class TestDigits {
	public static void main(String[] args) throws Exception {
		File dir = new File("C:/Users/valde/Desktop/digits"); // insert directory to network and test image here
		ObjectInputStream oin = new ObjectInputStream(new FileInputStream(new File(dir,"m2.ai")));
		Model model = (Model) oin.readObject();
		oin.close();
		Trainer t = new Trainer(model);
		BufferedImage inputImage = ImageIO.read(new File(dir, "test.png"));
		double[] input = new double[100];
		Arrays.fill(input, 1);
		for(int i = 0; i < 10; i++) {
			for(int j = 0; j < 10; j++) {
				input[i*10 + j] = (inputImage.getRGB(j, i) % 256)/256.0;
			}
		}
		double[] o = t.getOutput(input);
		System.out.println(Arrays.toString(o));
		for(int i = 0; i < 10; i++) {
			if(o[i] > 0.01) {
				System.out.println(Math.round(o[i]*1000)/10.0 + "% on " + i);
			}
		}
		
	}
}
