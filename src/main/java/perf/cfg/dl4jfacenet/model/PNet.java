package perf.cfg.dl4jfacenet.model;

import java.io.IOException;
import java.io.InputStream;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;

import perf.cfg.dl4jfacenet.model.custom.ActivationSoftMaxAxis;
import perf.cfg.dl4jfacenet.model.custom.PReLUNorm;

public class PNet extends AbstractModel {
	private long channels;

	public PNet(long channels) throws Exception {
		this.channels = channels;
	}

	public PNet() throws Exception {
		this(3);
	}

	protected ComputationGraph graph() throws Exception {
		String input = "input";
		GraphBuilder builder = new NeuralNetConfiguration.Builder().graphBuilder().addInputs(input)
				.addLayer("conv1",new ConvolutionLayer.Builder(3, 3)
						.nIn(channels)
						.nOut(10)
						.stride(1, 1)
						.convolutionMode(ConvolutionMode.Truncate)
						.activation(Activation.IDENTITY).build(),
						input)
				.addLayer("prelu1", new PReLUNorm.Builder()
						.nIn(10)
						.nOut(10).build(), "conv1")
				.addLayer("pool1", new SubsamplingLayer.Builder(2,2)
						.stride(2,2)
						.convolutionMode(ConvolutionMode.Same)
						.build(), "prelu1")
				.addLayer("conv2",new ConvolutionLayer.Builder(3, 3)
						.nIn(10)
						.nOut(16)
						.stride(1, 1)
						.convolutionMode(ConvolutionMode.Truncate)
						.activation(Activation.IDENTITY).build(),
						"pool1")
				.addLayer("prelu2", new PReLUNorm.Builder()
						.nIn(16)
						.nOut(16)
						.build(), "conv2")
				.addLayer("conv3",new ConvolutionLayer.Builder(3, 3)
						.nIn(16)
						.nOut(32)
						.stride(1, 1)
						.convolutionMode(ConvolutionMode.Truncate)
						.activation(Activation.IDENTITY).build(),
						"prelu2")
				.addLayer("prelu3", new PReLUNorm.Builder()
						.nIn(32)
						.nOut(32).build(), "conv3")
				.addLayer("conv4-1",new ConvolutionLayer.Builder(1,1)
						.nIn(32)
						.nOut(2)
						.stride(1, 1)
						.convolutionMode(ConvolutionMode.Same)
						.activation(new ActivationSoftMaxAxis(1)).build(),
						"prelu3")
				.addLayer("conv4-2",new ConvolutionLayer.Builder(1,1)
						.nIn(32)
						.nOut(4)
						.stride(1, 1)
						.convolutionMode(ConvolutionMode.Same)
						.activation(Activation.IDENTITY).build(),
						"prelu3")
				.setOutputs("conv4-1","conv4-2");
		ComputationGraph graph = new ComputationGraph(builder.build());
		graph.init();
		return graph;
	}

	@Override
	protected InputStream getModelDataInputStream() throws IOException {
		return Thread.currentThread().getContextClassLoader().getResourceAsStream("PNetData");
	}
}
