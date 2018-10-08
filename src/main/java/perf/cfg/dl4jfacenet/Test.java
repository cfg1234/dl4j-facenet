package perf.cfg.dl4jfacenet;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

import perf.cfg.dl4jfacenet.model.InceptionResNetV1;

public class Test {
	
	private static HashMap<String, List<ImgInfo>> imgInfoMap = new HashMap<String, List<ImgInfo>>();
	
	private static class CmdParam{
		@Parameter(description = "Test image path", required = true)
		String testImagePath;
	}
	
	private static void putImgInfo(ImgInfo info){
		List<ImgInfo> infoList = imgInfoMap.get(info.imageLabel);
		if(infoList == null){
			infoList = new ArrayList<ImgInfo>();
			imgInfoMap.put(info.imageLabel, infoList);
		}
		infoList.add(info);
	}
	
	private static INDArray prewhiten(INDArray x){
		double mean = x.mean().getDouble(0);
		double std = x.std().getDouble(0);
		double stdAdj = Math.max(std, 1.0/Math.sqrt(x.length()));
		return x.sub(mean).mul(1/stdAdj);
	}
	
	public static void main(String[] args) throws Exception {
		CmdParam param = new CmdParam();
		new JCommander(param).parse(args);
		
		File imagePath = new File(param.testImagePath);
		File[] subPaths = imagePath.listFiles();
		NativeImageLoader imageLoader = new NativeImageLoader();
		InceptionResNetV1 v1 = new InceptionResNetV1();
		ComputationGraph net = v1.getGraph();
		v1.loadWeightData();
		int fileSum = 0;
		for(File subPath:subPaths){
			if(!subPath.isDirectory()){
				continue;
			}
			String labelName = subPath.getName();
			for(File img:subPath.listFiles()){
				if(!img.isFile())
					continue;
				INDArray res = net.output(prewhiten(imageLoader.asMatrix(img)))[1];
				putImgInfo(new ImgInfo(res, labelName));
				System.out.println(String.format("Load image %s done.", img.getAbsolutePath()));
				fileSum++;
			}
		}
		System.out.println(fileSum);
		ArrayList<Double> sameLossList = new ArrayList<Double>();
		for(List<ImgInfo> infoList:imgInfoMap.values()){
			if(infoList.size() < 2) continue;
			for(int i = 0;i < infoList.size();i++){
				for(int j = i+1;j < infoList.size();j++){
					ImgInfo a = infoList.get(i),
							b = infoList.get(j);
					INDArray tmp = a.imgData.sub(b.imgData);
					tmp = tmp.mul(tmp.dup()).sum(1);
					sameLossList.add(tmp.getDouble(0));
				}
			}
		}
		
		System.out.printf("same loss size:%d\n", sameLossList.size());
		ArrayList<Double> diffLossList = new ArrayList<Double>();
		List<String> labelKey = new ArrayList<String>(imgInfoMap.keySet());
		top:for(int a = 0;a < labelKey.size();a++){
			List<ImgInfo> firstList = imgInfoMap.get(labelKey.get(a));
			for(int b = a + 1;b < labelKey.size();b++){
				List<ImgInfo> secondList = imgInfoMap.get(labelKey.get(b));
				for(int i = 0;i < firstList.size();i++){
					ImgInfo a1 = firstList.get(i);
					for(int j = 0;j < secondList.size();j++ ){
						ImgInfo a2 = secondList.get(j);
						INDArray tmp = a1.imgData.sub(a2.imgData);
						tmp = tmp.mul(tmp.dup()).sum(1);
						diffLossList.add(tmp.getDouble(0));
						if(diffLossList.size() >= sameLossList.size()){
							break top;
						}
					}
				}
			}
		}
		int subNum = sameLossList.size() - diffLossList.size();
		if(subNum > 0){
			for(int i = 0;i < subNum;i++){
				sameLossList.remove(sameLossList.size() - 1);
			}
		}
		
		int sampleSize = sameLossList.size();
		int trainSize = sampleSize * 9 / 10;
		double highestAccuracy = 0.0, highestThreshold = 0;
		for(double threshold = 0;threshold < 4;threshold += 0.01){
			long correct = 0, error = 0;
			for(int i = 0;i < trainSize;i++){
				if(sameLossList.get(i) <= threshold)
					correct++;
				else 
					error++;
				if(diffLossList.get(i) > threshold)
					correct++;
				else 
					error++;
			}
			double accuracy = correct * 100.0 / trainSize / 2;
			if(highestAccuracy < accuracy){
				highestAccuracy = accuracy;
				highestThreshold = threshold;
			}
			System.out.printf("threshold %2f, correct %d, wrong %d, accuracy %2f .\n",
					threshold, correct, error, accuracy);
		}
		System.out.println(String.format("highest accuracy:%2f, threshold=%2f", highestAccuracy, highestThreshold));
		int correct = 0,  error = 0, sum = 0;
		for(int i = trainSize;i < sampleSize;i++){
			if(sameLossList.get(i) <= highestThreshold)
				correct++;
			else 
				error++;
			if(diffLossList.get(i) > highestThreshold)
				correct++;
			else 
				error++;
			sum+=2;
		}
		double accuracy = correct * 100.0 / sum;
		System.out.printf("test, correct %d, wrong %d, accuracy %2f .\n",
				correct, error, accuracy);
	}
	
	private static class ImgInfo{
		INDArray imgData;
		String imageLabel;
		public ImgInfo(INDArray res, String imageLabel) {
			this.imgData = res;
			this.imageLabel = imageLabel;
		}
		public String toString(){
			return imageLabel;
		}
	}
}

