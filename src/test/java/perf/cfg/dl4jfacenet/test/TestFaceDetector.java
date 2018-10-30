package perf.cfg.dl4jfacenet.test;

import java.io.IOException;

import org.datavec.image.loader.NativeImageLoader;
import org.junit.BeforeClass;
import org.junit.Test;

import org.junit.Assert;
import perf.cfg.dl4jfacenet.FaceDetector;
import perf.cfg.dl4jfacenet.ResultBox;

/**
 * 
 */

/**
 * @author chenfengge
 *
 */
public class TestFaceDetector {
	
	private static FaceDetector detector;
	private static NativeImageLoader imageLoader = new NativeImageLoader();

	/**
	 * @throws java.lang.Exception
	 */
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		detector = new FaceDetector();
	}

	private static final double ALLOW_LOSS = 5;
	
	@Test
	public void testDetectFace() throws IOException {
		int[] std = {88, 61, 188, 193};
		ResultBox[] boxes = detector.detect_face(
				imageLoader.asMatrix(Thread.currentThread().getContextClassLoader()
						.getResourceAsStream("test.jpg")));
		Assert.assertEquals(1, boxes.length);
		TestUtil.assertAllowLoss("detect x1 ", ALLOW_LOSS, Math.abs(boxes[0].x1 - std[0]));
		TestUtil.assertAllowLoss("detect y1 ", ALLOW_LOSS, Math.abs(boxes[0].y1 - std[1]));
		TestUtil.assertAllowLoss("detect x2 ", ALLOW_LOSS, Math.abs(boxes[0].x2 - std[2]));
		TestUtil.assertAllowLoss("detect y2 ", ALLOW_LOSS, Math.abs(boxes[0].y2 - std[3]));
	}

}
