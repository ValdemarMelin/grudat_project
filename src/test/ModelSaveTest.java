package test;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import neuralnetwork.Model;
import neuralnetwork.model.DenseLayerDescriptor;
import neuralnetwork.model.LayerDescriptor;
import neuralnetwork.trainer.Trainer;

public class ModelSaveTest {
	
	/**
	 * Usage: java -ea ModelSaveTest
	 */
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		Model model = new Model(new LayerDescriptor[] {
				new DenseLayerDescriptor(1, 10),
				new DenseLayerDescriptor(10, 1)
		});
		Trainer t = new Trainer(model);
		t.randomizeWeights(1);
		File f = new File("filetest.ai");
		f.createNewFile();
		ObjectOutputStream dos = new ObjectOutputStream(new FileOutputStream(f));
		dos.writeObject(model);
		dos.close();
		ObjectInputStream dis = new ObjectInputStream(new FileInputStream(f));
		Model m2 = (Model)dis.readObject();
		dis.close();
		f.delete();
		assert m2.equals(model);
	}
}
