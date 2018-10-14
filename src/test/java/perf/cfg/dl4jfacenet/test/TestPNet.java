package perf.cfg.dl4jfacenet.test;

import static perf.cfg.dl4jfacenet.test.TestUtil.assertAllowLoss;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import perf.cfg.dl4jfacenet.model.PNet;

/**
 * 
 */

/**
 * @author chenfengge
 *
 */
public class TestPNet {
	
	private static ComputationGraph graph;

	/**
	 * @throws java.lang.Exception
	 */
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		PNet model = new PNet();
		model.init();
		model.loadWeightData();
		graph = model.getGraph();
	}

	private static final double ALLOW_LOSS = 0.00005;
	
	@Test
	public void baseTest() {
		INDArray testInput = Nd4j.ones(1, 3, 250,250);
		INDArray res[] = graph.output(testInput);
		assertAllowLoss("conv4-1 mean", ALLOW_LOSS, Math.abs(0.49999997 - res[0].mean().getDouble(0)));
		assertAllowLoss("conv4-1 sum", ALLOW_LOSS * res[0].length(), Math.abs(14399.999 - res[0].sum().getDouble(0)));
		assertAllowLoss("conv4-2 mean", ALLOW_LOSS, Math.abs(0.0026729628 - res[1].mean().getDouble(0)));
		assertAllowLoss("conv4-2 sum ", ALLOW_LOSS * res[1].length(), Math.abs(153.96266 - res[1].sum().getDouble(0)));
	}

}
