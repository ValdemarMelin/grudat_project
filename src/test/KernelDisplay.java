package test;

import java.awt.Color;
import java.awt.Graphics;

import javax.swing.JFrame;
import javax.swing.JPanel;

import neuralnetwork.MathUtilities;
import neuralnetwork.model.ConvLayerDescriptor;

/**
 * Displays trained kernel in a window.
 * @author valde
 *
 */
public class KernelDisplay extends JFrame {
	private static final long serialVersionUID = 1L;

	KernelDisplay(ConvLayerDescriptor dc, double[] weights) {
		setSize(800, 600);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		add(new JPanel() {
			private static final long serialVersionUID = 1L;

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
								g.fillRect(6*(x + (dc.size+4)*id), 6*(y + (dc.size+4)*od), 6, 6);
							}
						}
					}
				}
			}
		});
		setVisible(true);
	}
}
