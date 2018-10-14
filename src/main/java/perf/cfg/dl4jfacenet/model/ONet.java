package perf.cfg.dl4jfacenet.model;

import java.io.IOException;
import java.io.InputStream;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;

import perf.cfg.dl4jfacenet.model.custom.PReLUNorm;

public class ONet  extends AbstractModel {
	private long[] inputShape;

	public ONet(long... inputShape) throws Exception {
		this.inputShape = inputShape;
	}

	public ONet() throws Exception {
		this(new long[] { 48, 48, 3 });
	}


	protected ComputationGraph graph() throws Exception {
		String input = "input";
		GraphBuilder builder = new NeuralNetConfiguration.Builder().graphBuilder().addInputs(input)
				.setInputTypes(InputType.convolutional(inputShape[0], inputShape[1], inputShape[2]))
				.addLayer("conv1",new ConvolutionLayer.Builder(3, 3)
						.nOut(32)
						.stride(1, 1)
						.convolutionMode(ConvolutionMode.Truncate)
						.activation(Activation.IDENTITY).build(),
						input)
				.addLayer("prelu1", new PReLUNorm.Builder().build(), "conv1")
				.addLayer("pool1", new SubsamplingLayer.Builder(3,3)
						.stride(2,2)
						.convolutionMode(ConvolutionMode.Same)
						.build(), "prelu1")
				.addLayer("conv2",new ConvolutionLayer.Builder(3, 3)
						.nOut(64)
						.stride(1, 1)
						.convolutionMode(ConvolutionMode.Truncate)
						.activation(Activation.IDENTITY).build(),
						"pool1")
				.addLayer("prelu2", new PReLUNorm.Builder().build(), "conv2")
				.addLayer("pool2", new SubsamplingLayer.Builder(3,3)
						.stride(2,2)
						.convolutionMode(ConvolutionMode.Truncate)
						.build(), "prelu2")
				.addLayer("conv3",new ConvolutionLayer.Builder(3, 3)
						.nOut(64)
						.stride(1, 1)
						.convolutionMode(ConvolutionMode.Truncate)
						.activation(Activation.IDENTITY).build(),
						"pool2")
				.addLayer("prelu3", new PReLUNorm.Builder().build(), "conv3")
				.addLayer("pool3", new SubsamplingLayer.Builder(2,2)
						.stride(2,2)
						.convolutionMode(ConvolutionMode.Same)
						.build(), "prelu3")
				.addLayer("conv4",new ConvolutionLayer.Builder(2,2)
						.nOut(128)
						.stride(1, 1)
						.convolutionMode(ConvolutionMode.Truncate)
						.activation(Activation.IDENTITY).build(),
						"pool3")
				.addLayer("prelu4", new PReLUNorm.Builder().build(), "conv4")
				.addLayer("conv5", new DenseLayer.Builder().nOut(256)
						.activation(Activation.IDENTITY).build(), "prelu4")
				.addLayer("prelu5", new PReLUNorm.Builder().build(), "conv5")
				.addLayer("conv6-1", new DenseLayer.Builder().nOut(2)
						.activation(Activation.SOFTMAX).build(), "prelu5")
				.addLayer("conv6-2", new DenseLayer.Builder().nOut(4)
						.activation(Activation.IDENTITY).build(), "prelu5")
				.addLayer("conv6-3", new DenseLayer.Builder().nOut(10)
						.activation(Activation.IDENTITY).build(), "prelu5")
				.setOutputs("conv6-1", "conv6-2", "conv6-3");
		ComputationGraph graph = new ComputationGraph(builder.build());
		graph.init();
		return graph;
	}

	@Override
	protected InputStream getModelDataInputStream() throws IOException {
		return Thread.currentThread().getContextClassLoader().getResourceAsStream("ONetData");
	}
}
