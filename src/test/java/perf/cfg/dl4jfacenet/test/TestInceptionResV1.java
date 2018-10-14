package perf.cfg.dl4jfacenet.test;
import static perf.cfg.dl4jfacenet.test.TestUtil.assertAllowLoss;

import java.io.IOException;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import perf.cfg.dl4jfacenet.model.InceptionResNetV1;

/**
 * 
 */

/**
 * @author chenfengge
 *
 */
public class TestInceptionResV1 {
	
	private static ComputationGraph graph;

	/**
	 * @throws java.lang.Exception
	 */
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		InceptionResNetV1 model = new InceptionResNetV1();
		model.init();
		model.loadWeightData();
		graph = model.getGraph();
	}

	private static final double ALLOW_LOSS = 0.0000002;
	
	@Test
	public void baseTest() {
		INDArray testInput = Nd4j.ones(1, 3, 160,160);
		INDArray res[] = graph.output(testInput);
		assertAllowLoss("Logits sum ", ALLOW_LOSS * res[0].length(), Math.abs(-0.7470093 - res[0].sum().getDouble(0)));
		assertAllowLoss("Logits mean", ALLOW_LOSS, Math.abs(-1.6957443e-05 - res[0].mean().getDouble(0)));
		assertAllowLoss("Embeddings sum ", ALLOW_LOSS * res[1].length(), Math.abs(0.528594 - res[1].sum().getDouble(0)));
		assertAllowLoss("Embeddings mean", ALLOW_LOSS, Math.abs(0.0041296408 - res[1].mean().getDouble(0)));
	}
	
	private static INDArray prewhiten(INDArray x){
		double mean = x.mean().getDouble(0);
		double std = x.std().getDouble(0);
		double stdAdj = Math.max(std, 1.0/Math.sqrt(x.length()));
		return x.sub(mean).mul(1/stdAdj);
	}
	
	@Test
	public void imgTest() throws IOException {
		NativeImageLoader imageLoader = new NativeImageLoader();
		INDArray img1_0 = prewhiten(imageLoader.asMatrix(Thread.currentThread().getContextClassLoader()
				.getResourceAsStream("1_0.png"))),
				img1_1 = prewhiten(imageLoader.asMatrix(Thread.currentThread().getContextClassLoader()
						.getResourceAsStream("1_1.png"))),
				img2 = prewhiten(imageLoader.asMatrix(Thread.currentThread().getContextClassLoader()
						.getResourceAsStream("2.png")));
		INDArray factor1_0 = graph.output(img1_0)[1];
		INDArray factor1_1 = graph.output(img1_1)[1];
		INDArray factor2 = graph.output(img2)[1];
		double loss1 = calImgLoss(factor1_0, factor1_1),
				loss1_0_2 = calImgLoss(factor2, factor1_0),
				loss1_1_2 = calImgLoss(factor2, factor1_1);
		Assert.assertTrue(loss1 < 1);
		Assert.assertTrue(loss1_0_2 > 1);
		Assert.assertTrue(loss1_1_2 > 1);
	}
	
	private double calImgLoss(INDArray img1, INDArray img2){
		INDArray tmp = img1.sub(img2);
		tmp = tmp.mul(tmp).sum(1);
		return tmp.getDouble(0);
	}

}
