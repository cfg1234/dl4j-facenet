package perf.cfg.dl4jfacenet.test;

import static perf.cfg.dl4jfacenet.test.TestUtil.assertAllowLoss;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import perf.cfg.dl4jfacenet.model.ONet;

/**
 * 
 */

/**
 * @author chenfengge
 *
 */
public class TestONet {
	
	private static ComputationGraph graph;

	/**
	 * @throws java.lang.Exception
	 */
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		ONet model = new ONet();
		model.init();
		model.loadWeightData();
		graph = model.getGraph();
	}

	private static final double ALLOW_LOSS = 0.000005;
	
	@Test
	public void baseTest() {
		INDArray testInput = Nd4j.ones(1, 3, 48,48);
		INDArray res[] = graph.output(testInput);
		assertAllowLoss("conv6-1 mean", ALLOW_LOSS, Math.abs(0.5 - res[0].mean().getDouble(0)));
		assertAllowLoss("conv6-1 sum", ALLOW_LOSS * res[0].length(), Math.abs(1.0 - res[0].sum().getDouble(0)));
		assertAllowLoss("conv6-2 mean", ALLOW_LOSS, Math.abs(-0.016578328 - res[1].mean().getDouble(0)));
		assertAllowLoss("conv6-2 sum ", ALLOW_LOSS * res[1].length(), Math.abs(-0.06631331 - res[1].sum().getDouble(0)));
		assertAllowLoss("conv6-3 mean", ALLOW_LOSS, Math.abs(0.50267076 - res[2].mean().getDouble(0)));
		assertAllowLoss("conv6-3 sum ", ALLOW_LOSS * res[2].length(), Math.abs(5.0267076 - res[2].sum().getDouble(0)));
	}

}
