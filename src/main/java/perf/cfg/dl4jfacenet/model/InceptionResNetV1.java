package perf.cfg.dl4jfacenet.model;

import static org.deeplearning4j.nn.conf.ConvolutionMode.Truncate;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import perf.cfg.dl4jfacenet.model.custom.ActivationLinear;
import perf.cfg.dl4jfacenet.model.custom.ActivationReverse;

public class InceptionResNetV1 {
	private ComputationGraph graph;
	private long[] inputShape = new long[] { 160, 160, 3 };

	public InceptionResNetV1(long... inputShape) throws Exception {
		this.inputShape = inputShape;
		init();
	}

	public InceptionResNetV1() throws Exception {
		init();
	}
	
	public ComputationGraph getGraph(){
		return graph;
	}

	private void init() throws Exception {
		String input = "input";
		GraphBuilder builder = new NeuralNetConfiguration.Builder().graphBuilder().addInputs(input)
				.setInputTypes(InputType.convolutional(inputShape[0], inputShape[1], inputShape[2]));
		GraphBuilderHelper helper = new GraphBuilderHelper(builder);
		helper.lastLayer = input;
		helper.layerConfMap.put(input, new LayerConf(input, (int) inputShape[2]));
		helper.addLayerAndBatchNormBehind("Conv2d_1a_3x3", defConv(32, 3, 2).convolutionMode(Truncate));
		helper.addLayerAndBatchNormBehind("Conv2d_2a_3x3", defConv(32, 3).convolutionMode(Truncate));
		helper.addLayerAndBatchNormBehind("Conv2d_2b_3x3", defConv(64, 3));
		helper.addLayerBehind("MaxPool_3a_3x3", defPool(PoolingType.MAX, 3, 2).convolutionMode(Truncate));
		helper.addLayerAndBatchNormBehind("Conv2d_3b_1x1", defConv(80, 1).convolutionMode(Truncate));
		helper.addLayerAndBatchNormBehind("Conv2d_4a_3x3", defConv(192, 3).convolutionMode(Truncate));
		helper.addLayerAndBatchNormBehind("Conv2d_4b_3x3", defConv(256, 3, 2).convolutionMode(Truncate));
		for (int i = 0; i < 5; i++) {
			block35(helper, "block35_" + i + "/", 0.17);
		}
		reduction_a(helper, 192, 192, 256, 384);
		for (int i = 0; i < 10; i++) {
			block17(helper, "block17_" + i + "/", 0.1);
		}
		reduction_b(helper);
		for (int i = 0; i < 5; i++) {
			block8(helper, true, "block8_" + i, 0.2);
		}
		block8(helper, false, "block8_final", 1);
		helper.addLayerBehind("avg_pool", defPool(PoolingType.AVG, 3).convolutionMode(Truncate));
		helper.addLayerBehind("Dropout", new DropoutLayer.Builder(0.8));
		helper.addLayerBehind("reverse", new ActivationLayer.Builder(new ActivationReverse()));
		helper.addLayerAndBatchNormBehind("Bottleneck", defDense(128), Activation.IDENTITY);
		helper.addLayerBehind("logits", defDense(44052).hasBias(true));
		helper.addVertex("embeddings", new L2NormalizeVertex(new int[] { 1 }, 1e-10), 0, toActName("Bottleneck"));
		builder.setOutputs("logits", "embeddings");
		graph = new ComputationGraph(builder.build());
		graph.init();
	}
	
	private static class MultiResouceInputStream{
		private String resourceName;
		private InputStream in = null;
		private int currIndex = 0;

		public MultiResouceInputStream(String resourceName) {
			this.resourceName = resourceName;
			String currResource = resourceName + "_" + (currIndex++);
			in = Thread.currentThread().getContextClassLoader()
					.getResourceAsStream(currResource); 
		}
		
		public int read(byte[] b, int off, int len) throws IOException{
			if(in == null){
				return -1;
			}
			int ret = in.read(b, off, len);
			if(ret == len){
				return ret;
			} else if(ret < 0){
				in.close();
				in = Thread.currentThread().getContextClassLoader()
						.getResourceAsStream(resourceName + "_" + (currIndex++)); 
				return read(b, off, len);
			} else {
				int tmp = read(b, off + ret, len - ret);
				if(tmp < 0){
					tmp = 0;
				}
				return tmp + ret;
			}
		}
		
		public int read(byte[] b) throws IOException{
			return read(b, 0, b.length);
		}
		
		public void close() throws IOException{
			if(in != null){
				in.close();
			}
		}
	}

	public void loadWeightData() throws IOException {
		org.deeplearning4j.nn.api.Layer[] layers = graph.getLayers();
		MultiResouceInputStream fis = new MultiResouceInputStream("InceptionResNetV1Data");
		byte[] buf = new byte[4];
		try{
			for (org.deeplearning4j.nn.api.Layer l:layers) {
				int nParams = l.numParams();
				if(nParams == 0) continue;
				float[] data = new float[nParams];
				for(int i = 0;i < nParams;i++){
					fis.read(buf);
					data[i] = bytes2Float(buf);
				}
				l.setParams(Nd4j.create(data));
			}
		} finally {
			fis.close();
		}
	}

	public static float bytes2Float(byte[] arr) {
		int value = 0;
		for (int i = 0; i < 4; i++) {
			value |= ((int) (arr[i] & 0xff)) << (8 * i);
		}
		return Float.intBitsToFloat(value);
	}

	private static String nameLayer(String original, String i) {
		return String.format("(%s)%s", i, original);
	}

	private static void block35(GraphBuilderHelper helper, String prefix, final double scale) throws Exception {
		String input = helper.lastLayer;
		String tower_conv = prefix + "/Branch_0/Conv2d_1x1";
		String tower_conv1_0 = prefix + "/Branch_1/Conv2d_0a_1x1";
		String tower_conv1_1 = prefix + "/Branch_1/Conv2d_0b_3x3";
		String tower_conv2_0 = prefix + "/Branch_2/Conv2d_0a_1x1";
		String tower_conv2_1 = prefix + "/Branch_2/Conv2d_0b_3x3";
		String tower_conv2_2 = prefix + "/Branch_2/Conv2d_0c_3x3";
		String mixed = prefix + "/mixed";
		String up = prefix + "/Conv2d_1x1";
		String scaleName = prefix + "/scale";
		String add = prefix + "/add";
		String relu = prefix + "/relu";
		helper.addLayerAndBatchNorm(tower_conv, defConv(32, 1), input);
		helper.addLayerAndBatchNorm(tower_conv1_0, defConv(32, 1), input);
		helper.addLayerAndBatchNormBehind(tower_conv1_1, defConv(32, 3));
		helper.addLayerAndBatchNorm(tower_conv2_0, defConv(32, 1), input);
		helper.addLayerAndBatchNormBehind(tower_conv2_1, defConv(32, 3));
		helper.addLayerAndBatchNormBehind(tower_conv2_2, defConv(32, 3));
		String[] merges = new String[] { toActName(tower_conv2_2), toActName(tower_conv1_1), toActName(tower_conv) };
		helper.addVertex(mixed, new MergeVertex(), helper.getOutput(merges), merges);
		helper.addLayer(up, defConv(helper.getOutput(input), 1).hasBias(true), mixed);
		helper.addLayerBehind(scaleName, new ActivationLayer.Builder(new ActivationLinear(scale)));
		helper.addVertex(add, new ElementWiseVertex(ElementWiseVertex.Op.Add), helper.getOutput(input), input,
				scaleName);
		helper.addLayer(relu, new ActivationLayer.Builder().activation(Activation.RELU), add);
	}

	private static void block17(GraphBuilderHelper helper, String i, final double scale) throws Exception {
		String b0 = nameLayer("b0", i), b1_0 = nameLayer("b1_0", i), b1_1 = nameLayer("b1_1", i),
				b1_2 = nameLayer("b2_0", i), mixed = nameLayer("mixed", i), up = nameLayer("up", i),
				scaleName = nameLayer("scaleName", i), add = nameLayer("add", i),
				activation = nameLayer("activation", i);
		String input = helper.lastLayer;
		helper.addLayerAndBatchNorm(b0, defConv(128, 1), input);
		helper.addLayerAndBatchNorm(b1_0, defConv(128, 1), input);
		helper.addLayerAndBatchNormBehind(b1_1, defConv(128, 1).kernelSize(1, 7));
		helper.addLayerAndBatchNormBehind(b1_2, defConv(128, 1).kernelSize(7, 1));
		String[] merges = new String[] { toActName(b1_2), toActName(b0) };
		helper.addVertex(mixed, new MergeVertex(), helper.getOutput(merges), merges);
		helper.addLayer(up, defConv(helper.getOutput(input), 1).hasBias(true), mixed);
		helper.addLayerBehind(scaleName, new ActivationLayer.Builder(new ActivationLinear(scale)));
		helper.addVertex(add, new ElementWiseVertex(ElementWiseVertex.Op.Add), helper.getOutput(input), input,
				scaleName);
		helper.addLayer(activation, new ActivationLayer.Builder().activation(Activation.RELU), add);
	}

	private static void block8(GraphBuilderHelper helper, boolean activateFunc, String i, final double scale)
			throws Exception {
		String b0 = nameLayer("b0", i), b1_0 = nameLayer("b1_0", i), b1_1 = nameLayer("b1_1", i),
				b1_2 = nameLayer("b2_0", i), mixed = nameLayer("mixed", i), up = nameLayer("up", i),
				scaleName = nameLayer("scaleName", i), add = nameLayer("add", i),
				activation = nameLayer("activation", i);
		String input = helper.lastLayer;
		helper.addLayerAndBatchNormBehind(b0, defConv(192, 1));
		helper.addLayerAndBatchNorm(b1_0, defConv(192, 1), input);
		helper.addLayerAndBatchNormBehind(b1_1, defConv(192, 1).kernelSize(1, 3));
		helper.addLayerAndBatchNormBehind(b1_2, defConv(192, 1).kernelSize(3, 1));
		String[] merges = new String[] { toActName(b1_2), toActName(b0) };
		helper.addVertex(mixed, new MergeVertex(), helper.getOutput(merges), merges);
		helper.addLayerBehind(up, defConv(helper.getOutput(input), 1).hasBias(true));
		helper.addLayerBehind(scaleName, new ActivationLayer.Builder(new ActivationLinear(scale)));
		helper.addVertex(add, new ElementWiseVertex(ElementWiseVertex.Op.Add), helper.getOutput(input), input,
				scaleName);
		Activation act = activateFunc ? Activation.RELU : Activation.IDENTITY;
		helper.addLayer(activation, new ActivationLayer.Builder().activation(act), add);
	}

	private static void reduction_a(GraphBuilderHelper helper, int k, int l, int m, int n) throws Exception {
		String tower_conv = nameLayer("tower_conv", "a"), tower_conv1_0 = nameLayer("tower_conv1_0", "a"),
				tower_conv1_1 = nameLayer("tower_conv1_1", "a"), tower_conv1_2 = nameLayer("tower_conv1_2", "a"),
				tower_pool = nameLayer("tower_pool", "a"), res = nameLayer("res", "a");
		String input = helper.lastLayer;
		helper.addLayerAndBatchNormBehind(tower_conv, defConv(n, 3, 2).convolutionMode(Truncate));
		helper.addLayerAndBatchNorm(tower_conv1_0, defConv(k, 1), input);
		helper.addLayerAndBatchNormBehind(tower_conv1_1, defConv(l, 3));
		helper.addLayerAndBatchNormBehind(tower_conv1_2, defConv(m, 3, 2).convolutionMode(Truncate));
		helper.addLayer(tower_pool, defPool(PoolingType.MAX, 3, 2).convolutionMode(Truncate), input);
		String[] merges = new String[] { tower_pool, toActName(tower_conv1_2), toActName(tower_conv) };
		helper.addVertex(res, new MergeVertex(), helper.getOutput(merges), merges);
	}

	private static void reduction_b(GraphBuilderHelper helper) throws Exception {
		String tower_conv = nameLayer("tower_conv", "b"), tower_conv_1 = nameLayer("tower_conv_1", "b"),
				tower_conv1 = nameLayer("tower_conv1_0", "b"), tower_conv1_1 = nameLayer("tower_conv1_1", "b"),
				tower_conv2 = nameLayer("tower_conv2", "b"), tower_conv2_1 = nameLayer("tower_conv2_1", "b"),
				tower_conv2_2 = nameLayer("tower_conv2_2", "b"), tower_pool = nameLayer("tower_pool", "b"),
				res = nameLayer("res", "b");
		String input = helper.lastLayer;
		helper.addLayerAndBatchNormBehind(tower_conv, defConv(256, 1));
		helper.addLayerAndBatchNormBehind(tower_conv_1, defConv(384, 3, 2).convolutionMode(Truncate));
		helper.addLayerAndBatchNorm(tower_conv1, defConv(256, 1), input);
		helper.addLayerAndBatchNormBehind(tower_conv1_1, defConv(256, 3, 2).convolutionMode(Truncate));
		helper.addLayerAndBatchNorm(tower_conv2, defConv(256, 1), input);
		helper.addLayerAndBatchNormBehind(tower_conv2_1, defConv(256, 3));
		helper.addLayerAndBatchNormBehind(tower_conv2_2, defConv(256, 3, 2).convolutionMode(Truncate));
		helper.addLayer(tower_pool, defPool(PoolingType.MAX, 3, 2).convolutionMode(Truncate), input);
		String[] merges = new String[] { tower_pool, toActName(tower_conv2_2), toActName(tower_conv1_1),
				toActName(tower_conv_1) };
		helper.addVertex(res, new MergeVertex(), helper.getOutput(merges), merges);
	}

	private static ConvolutionLayer.Builder defConv(int output, int kernelSize, int stride) {
		double weight_decay = 0;
		return new ConvolutionLayer.Builder(kernelSize, kernelSize).weightInit(WeightInit.DISTRIBUTION)
				.dist(new TruncatedNormalDistribution(0, 0.1)).l2(weight_decay).stride(stride, stride).nOut(output)
				.convolutionMode(ConvolutionMode.Same).activation(Activation.IDENTITY).hasBias(false);
	}

	private static DenseLayer.Builder defDense(int output) {
		double weight_decay = 0;
		return new DenseLayer.Builder().weightInit(WeightInit.DISTRIBUTION)
				.dist(new TruncatedNormalDistribution(0, 0.1)).l2(weight_decay).nOut(output)
				.activation(Activation.IDENTITY).hasBias(false);
	}

	private static ConvolutionLayer.Builder defConv(int output, int kernelSize) {
		return defConv(output, kernelSize, 1);
	}

	private static BatchNormalization.Builder defBatchNorm() {
		return new BatchNormalization.Builder(false).decay(0.995).eps(0.001);
	}

	private static SubsamplingLayer.Builder defPool(PoolingType type, int kernelSize, int stride) {
		return new SubsamplingLayer.Builder(type).kernelSize(kernelSize, kernelSize).stride(stride, stride);
	}

	private static SubsamplingLayer.Builder defPool(PoolingType type, int kernelSize) {
		return defPool(type, kernelSize, 1);
	}

	private static String toBatchNormName(String name) {
		return name + "/batch_norm";
	}

	private static String toActName(String name) {
		return name + "/act";
	}

	private static class GraphBuilderHelper {
		GraphBuilder builder;
		String lastLayer = null;
		Map<String, LayerConf> layerConfMap = new HashMap<String, LayerConf>();
		Field outField;

		public GraphBuilderHelper(GraphBuilder builder) throws Exception {
			super();
			this.builder = builder;
			outField = FeedForwardLayer.Builder.class.getDeclaredField("nOut");
			outField.setAccessible(true);
		}

		int getOutput(String layerName) {
			LayerConf conf = layerConfMap.get(layerName);
			if (conf == null) {
				throw new RuntimeException("unknown layer:" + layerName);
			}
			if (conf.nOut > 0) {
				return conf.nOut;
			}
			return getOutput(conf.input);
		}

		int getOutput(String[] layerNames) {
			int sum = 0;
			for (String input : layerNames) {
				sum += getOutput(input);
			}
			return sum;
		}

		void addLayer(String name, @SuppressWarnings("rawtypes") Layer.Builder layer, String... input)
				throws Exception {
			int nOut = 0;
			if (layer instanceof FeedForwardLayer.Builder) {
				nOut = outField.getInt(layer);
			}
			builder.addLayer(name, layer.build(), input);
			lastLayer = name;
			layerConfMap.put(name, new LayerConf(name, nOut, input));
		}

		void addLayerBehind(String name, @SuppressWarnings("rawtypes") Layer.Builder layer) throws Exception {
			if (lastLayer == null) {
				throw new RuntimeException("no last layer");
			}
			addLayer(name, layer, lastLayer);
		}

		void addLayerAndBatchNorm(String name, @SuppressWarnings("rawtypes") Layer.Builder layer, Activation act,
				String... input) throws Exception {
			addLayer(name, layer, input);
			addLayerBehind(toBatchNormName(name), defBatchNorm());
			addLayerBehind(toActName(name), new ActivationLayer.Builder().activation(act));
		}

		void addLayerAndBatchNorm(String name, @SuppressWarnings("rawtypes") Layer.Builder layer, String... input)
				throws Exception {
			addLayerAndBatchNorm(name, layer, Activation.RELU, input);
		}

		void addLayerAndBatchNormBehind(String name, @SuppressWarnings("rawtypes") Layer.Builder layer, Activation act)
				throws Exception {
			addLayerAndBatchNorm(name, layer, act, lastLayer);
		}

		void addLayerAndBatchNormBehind(String name, @SuppressWarnings("rawtypes") Layer.Builder layer)
				throws Exception {
			addLayerAndBatchNormBehind(name, layer, Activation.RELU);
		}

		void addVertex(String vertexName, GraphVertex vertex, int nOut, String... vertexInputs) {
			builder.addVertex(vertexName, vertex, vertexInputs);
			lastLayer = vertexName;
			layerConfMap.put(vertexName, new LayerConf(vertexName, nOut, vertexInputs));
		}

		void addVertexBehind(String vertexName, GraphVertex vertex, int nOut, String... vertexInputs) {
			if (lastLayer == null) {
				throw new RuntimeException("no last layer");
			}
			addVertex(vertexName, vertex, nOut, lastLayer);
		}
	}

	private static class LayerConf {
		int nOut;
		String[] input;

		public LayerConf(String name, int nOut, String... input) {
			super();
			this.nOut = nOut;
			this.input = input;
		}
	}
}
