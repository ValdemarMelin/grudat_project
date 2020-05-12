package test;

import java.io.File;
import java.util.Scanner;

import neuralnetwork.Model;
import neuralnetwork.model.DenseLayerDescriptor;
import neuralnetwork.model.LayerDescriptor;
import neuralnetwork.trainer.Trainer;

/**
 * Warning: not finished.
 * 
 * Console for training.
 * @author Valdemar Melin
 *
 */
public class ConsoleTrainer {
	private Trainer trainer;
	private Thread trainingThread;
	private double[][] inputData;
	private double[][] outputData;
	private File saveFile;
	
	private volatile int remainingIterations;
	private volatile double step;
	private volatile boolean stopTraining;
	
	public ConsoleTrainer(Model model, File file, double[][] inputData, double[][] outputData) {
		this.saveFile = file;
		this.inputData = inputData;
		this.outputData = outputData;
		trainer = new Trainer(model);
		trainingThread = new Thread(this::runThread);
	}
	
	public void runConsole() {
		stopTraining = false;
		trainingThread.start();
		Scanner in = new Scanner(System.in);
		while(trainingThread.isAlive()) {
			System.out.print("trainer >");
			String[] line = in.nextLine().split(" ");
			synchronized(this) {
				switch(line[0].toLowerCase().trim()) {
				case "":
					break;
				case "load":
					break;
				case "stop":
					remainingIterations = 0;
					break;
				case "train":
					if(remainingIterations > 0) {
						System.out.println("Training already in progress");
					}
					else {
						remainingIterations = Integer.valueOf(line[1]);
						step = Double.valueOf(line[2]);
						notify();
					}
					break;
				case "progress":
					System.out.println("Remaining iterations: " + remainingIterations);
					break;
				case "save":
					if(saveFile == null) {
						in.nextLine();
						System.out.println("Enter file:");
						saveFile = new File(in.nextLine().trim());
					}
					save();
					break;
				case "saveto":
					saveFile = new File(in.nextLine().trim());
					save();
					break;
				case "exit":
					stopTraining = true;
					remainingIterations = 0;
					while(true) {
						System.out.println("Save before exit? [y/n]");
						String s = in.next().trim().toLowerCase();
						in.nextLine();
						if(s.equals("y")) {
							if(saveFile == null) {
								System.out.println("Enter file:");
								saveFile = new File(in.nextLine().trim());
							}
							save();
						}
						else if(s.equals("n")) {
							break;
						}
						else {
							System.out.println("Options: \"y\",\"n\"");
						}
					}
					break;
				case "cost":
					System.out.println(trainer.cost(inputData, outputData));
					break;
				default:
					System.out.println("Error: " + line[0] + ": unknown command");
					break;
				}
			}
		}
		in.close();
	}
	
	private void save() {
		
	}
	
	private void runThread() {
		while(!stopTraining) {
			synchronized(this) {
				try {
					wait();
				} catch(Exception e) {
					e.printStackTrace();
				}
			}
			while(remainingIterations > 0) {
				synchronized(this) {
					trainer.eval(inputData, outputData);
					trainer.step(step);
					remainingIterations--;
				}
			}
		}
	}
	
	public static void main(String[] args) {
		double[][] ins = new double[][] {new double[] {-1},new double[] {1}};
		double[][] outs = new double[][] {new double[] {0}, new double[] {1}};
		Model m = new Model(new LayerDescriptor[] {
				new DenseLayerDescriptor(1, 1)
		});
		new ConsoleTrainer(m, null,ins, outs).runConsole();
	}
}
