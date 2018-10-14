package perf.cfg.dl4jfacenet.test;

import static perf.cfg.dl4jfacenet.test.TestUtil.assertAllowLoss;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import perf.cfg.dl4jfacenet.model.RNet;

/**
 * 
 */

/**
 * @author chenfengge
 *
 */
public class TestRNet {
	
	private static ComputationGraph graph;

	/**
	 * @throws java.lang.Exception
	 */
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		RNet model = new RNet();
		model.init();
		model.loadWeightData();
		graph = model.getGraph();
	}

	private static final double ALLOW_LOSS = 0.000005;
	
	@Test
	public void baseTest() {
		INDArray testInput = Nd4j.ones(1, 3, 24,24);
		INDArray res[] = graph.output(testInput);
		assertAllowLoss("conv5-1 mean", ALLOW_LOSS, Math.abs(0.5 - res[0].mean().getDouble(0)));
		assertAllowLoss("conv5-1 sum", ALLOW_LOSS * res[0].length(), Math.abs(1.0 - res[0].sum().getDouble(0)));
		assertAllowLoss("conv5-2 mean", ALLOW_LOSS, Math.abs(0.025646359 - res[1].mean().getDouble(0)));
		assertAllowLoss("conv5-2 sum ", ALLOW_LOSS * res[1].length(), Math.abs(0.102585435 - res[1].sum().getDouble(0)));
	}

}
