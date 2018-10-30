package perf.cfg.dl4jfacenet;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Scanner;

import javax.imageio.ImageIO;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

import perf.cfg.dl4jfacenet.model.InceptionResNetV1;

public class FaceDetection {
	
	private static final Logger log = LoggerFactory.getLogger(FaceDetection.class);
	private static final NativeImageLoader loader = new NativeImageLoader();
	private static final HashMap<String, INDArray> imageLibMap = new HashMap<String, INDArray>();
	private static FaceDetector detector;
	private static ComputationGraph graph;
	
	private static class CmdParam{
		@Parameter(description = "face libirary path", required = true)
		String libPath;
		
		@Parameter(names = {"-t", "--threshold"}, description = "compare loss threshold", required = false)
		double threshold = 1.1;
	}

	public static void main(String[] args) throws Exception {
		CmdParam param = new CmdParam();
		new JCommander(param).parse(args);
		detector = new FaceDetector();
		InceptionResNetV1 v1 = new InceptionResNetV1();
		v1.init();
		v1.loadWeightData();
		graph = v1.getGraph();
		File libPath = new File(param.libPath);
		if(!libPath.exists() || !libPath.isDirectory()) {
			throw new RuntimeException("path " + libPath + "is not directory!");
		}
		log.info("loading face library from directory {}, threshold is {}", libPath, param.threshold);
		for(File file:libPath.listFiles()) {
			if(!file.isFile()) {
				log.info("skipped directory:{}", file);
				continue;
			}
			try {
				String label = file.getName();
				int labelIdx = label.lastIndexOf('.');
				if(labelIdx != -1) {
					label = label.substring(0, labelIdx);
				}
				log.info("Loading image {}......", file);
				INDArray factor = getFaceFactor(file);
				if(factor == null) {
					continue;
				}

				imageLibMap.put(label, factor);
				log.info("Face of {} loaded.", label);
				if(log.isDebugEnabled()) {
					log.debug("factor of {} is {}", label, factor);
				}
			} catch (IOException e) {
				log.warn("Exception occured while loading img file:" + file, e);
			}
		}
		log.info("load faces complete.");
		System.out.println("start detect");
		Scanner sc = new Scanner(System.in);
		while(true) {
			try {
				System.out.print("Insert image path:");
				File file = new File(sc.nextLine());
				INDArray factor = getFaceFactor(file);
				if(factor == null) {
					System.out.println("error:cannot detect face.");
					continue;
				}
				double minVal = Double.MAX_VALUE;
				String label = "none";
				for(Entry<String, INDArray> entry:imageLibMap.entrySet()) {
					INDArray tmp = factor.sub(entry.getValue());
					tmp = tmp.mul(tmp).sum(1);
					double tmpVal = tmp.getDouble(0);
					if(log.isDebugEnabled()) {
						log.debug("current factor is {}", factor);
						log.debug("similarity with {} is {}", entry.getKey(), tmp);
					}
					if(tmpVal < minVal) {
						minVal = tmpVal;
						label = entry.getKey();
					}
				}
				
				if(minVal < param.threshold) {
					System.out.println("this is " + label + "(" + minVal + ").");
				} else {
					System.out.println("cannot recognize this person, but the similar one is "
							+ label + "(" + minVal + ").");
				}

			}catch(Exception e) {
				e.printStackTrace();
			}	
		}
	}
	
	private static INDArray getFaceFactor(File img) throws IOException {
		INDArray srcImg = loader.asMatrix(img);
		INDArray[] detection = detector.getFaceImage(srcImg, 160, 160);
		if(detection == null) {
			log.warn("no face detected in image file:{}", img);
			return null;
		}
		if(detection.length > 1) {
			log.warn("{} faces detected in image file:{}, the first detected face will be used.", 
					detection.length, img);
		}
		if(log.isDebugEnabled()) {
			ImageIO.write(detector.toImage(detection[0]), "jpg", new File(img.getName()));
		}
		INDArray output[] = graph.output(InceptionResNetV1.prewhiten(detection[0]));
		return output[1];
	}

}
