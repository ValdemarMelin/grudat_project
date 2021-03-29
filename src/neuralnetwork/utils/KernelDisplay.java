package neuralnetwork.utils;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;

import neuralnetwork.MathUtilities;
import neuralnetwork.Model;
import neuralnetwork.model.ConvLayerDescriptor;

/**
 * Displays trained kernels of a network.
 * @author Valdemar Melin
 *
 */
public class KernelDisplay extends JPanel {
	private static final long serialVersionUID = 1L;

	
	private class KernelDisplayPane extends JPanel {
		private static final long serialVersionUID = 1L;
		
		private final ConvLayerDescriptor dc;
		private final double[] weights;
		private final int factor;
		private final int stride;
		
		public KernelDisplayPane(ConvLayerDescriptor dc, double[] weights) {
			this.dc = dc;
			this.weights = weights;
			this.factor = 6;
			this.stride = 4;
			Dimension size = new Dimension((dc.size+stride)*dc.inD*factor, (dc.size+stride)*dc.depth*factor);
			setSize(size);
			setMinimumSize(size);
			setPreferredSize(size);
		}

		@Override
		public void paintComponent(Graphics g) {
			super.paintComponent(g);
			for(int id = 0; id < dc.inD; id++) {
				for(int od = 0; od < dc.depth; od++) {
					for(int x = 0; x < dc.size; x++) {
						for(int y = 0; y < dc.size; y++) {
							double k = MathUtilities.f(weights[x + y*dc.size + id*dc.size*dc.size + od*dc.size*dc.size*dc.inD]);
							int cc = (int)(255*k);
							g.setColor(new Color(cc,cc,cc));
							g.fillRect(factor*(x + (dc.size+stride)*id), factor*(y + (dc.size+stride)*od), 6, 6);
						}
					}
				}
			}
		}
	}
	
	/**
	 * Create a window containing a KernelDisplay of the model.
	 * @param model the model.
	 * @return the window.
	 */
	public static JFrame window(Model model) {
		JFrame frame = new JFrame();
		frame.setTitle("Convolution layer viewer");
		frame.setSize(500, 500);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		KernelDisplay disp = new KernelDisplay(model);
		frame.getContentPane().add(disp);
		frame.setVisible(true);
		return frame;
	}
	
	/**
	 * Create a new KernelDisplay panel for the model.
	 * @param model the model.
	 */
	public KernelDisplay(Model model) {
		setLayout(new BorderLayout());
		JTabbedPane tabbed = new JTabbedPane();
		for(int l = 0; l < model.getLayerCount(); l++) {
			if(!(model.getLayerDescriptor(l) instanceof ConvLayerDescriptor)) continue;
			ConvLayerDescriptor dc = (ConvLayerDescriptor) model.getLayerDescriptor(l);
			KernelDisplayPane pane = new KernelDisplayPane(dc, model.params[l]);
			JScrollPane scroll = new JScrollPane(pane, JScrollPane.VERTICAL_SCROLLBAR_ALWAYS, JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
			scroll.setViewportView(pane);
			tabbed.add("Layer " + l, scroll);
		}
		add(tabbed, BorderLayout.CENTER);
	}
}
