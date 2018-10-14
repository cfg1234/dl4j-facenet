package perf.cfg.dl4jfacenet.model;

import java.io.IOException;
import java.io.InputStream;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.factory.Nd4j;

public abstract class AbstractModel {
	private ComputationGraph graph;

	public ComputationGraph getGraph() {
		return graph;
	}

	protected abstract InputStream getModelDataInputStream() throws IOException;
	
	public void init() throws Exception{
		graph = graph();
	}
	
	protected abstract ComputationGraph graph() throws Exception;

	public void loadWeightData() throws IOException {
		org.deeplearning4j.nn.api.Layer[] layers = graph.getLayers();
		InputStream in = getModelDataInputStream();
		byte[] buf = new byte[4];
		try {
			for (org.deeplearning4j.nn.api.Layer l : layers) {
				int nParams = l.numParams();
				if (nParams == 0)
					continue;
				float[] data = new float[nParams];
				for (int i = 0; i < nParams; i++) {
					in.read(buf);
					data[i] = bytes2Float(buf);
				}
				l.setParams(Nd4j.create(data));
			}
		} finally {
			in.close();
		}
	}

	public static float bytes2Float(byte[] arr) {
		int value = 0;
		for (int i = 0; i < 4; i++) {
			value |= ((int) (arr[i] & 0xff)) << (8 * i);
		}
		return Float.intBitsToFloat(value);
	}
}
