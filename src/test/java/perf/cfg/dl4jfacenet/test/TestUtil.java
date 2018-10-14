package perf.cfg.dl4jfacenet.test;

import org.junit.Assert;

public class TestUtil {
	
	public static void assertAllowLoss(String title, double allowLoss, double actualLoss){
		Assert.assertTrue(title + " allow loss is " + allowLoss + ", actual loss is " + actualLoss, actualLoss < allowLoss);
	}
}
