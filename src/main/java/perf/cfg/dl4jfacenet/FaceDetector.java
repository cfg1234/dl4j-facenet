package perf.cfg.dl4jfacenet;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Arrays;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import perf.cfg.dl4jfacenet.model.AbstractModel;
import perf.cfg.dl4jfacenet.model.ONet;
import perf.cfg.dl4jfacenet.model.PNet;
import perf.cfg.dl4jfacenet.model.RNet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FaceDetector {
	
	private static final Logger log = LoggerFactory.getLogger(FaceDetector.class);
	private ComputationGraph pnet, rnet, onet;
	private double thresholds[] = { 0.6, 0.7, 0.7 };
	private int minSize = 20;
	private double factor = 0.709;
	
	public FaceDetector() throws Exception {
		AbstractModel models[] = new AbstractModel[] { new PNet(), new RNet(), new ONet() };
		for (AbstractModel m : models) {
			m.init();
			m.loadWeightData();
		}
		pnet = models[0].getGraph();
		rnet = models[1].getGraph();
		onet = models[2].getGraph();
	}
	
	
	
	public FaceDetector(double[] thresholds, int minSize, double factor) throws Exception {
		this();
		this.thresholds = thresholds;
		this.minSize = minSize;
		this.factor = factor;
	}



	public INDArray[] getFaceImage(INDArray img, int width, int height) {
		ResultBox[] boxes = detect_face(img);
		if(boxes == null) {
			return null;
		}
		INDArray[] ret = new INDArray[boxes.length];
		for(int i = 0;i < ret.length;i++) {
			ret[i] = imresample(img.get(all(), all(), interval(boxes[0].y1,boxes[0].y2), interval(boxes[0].x1,boxes[0].x2)).dup(), height, width);
		}
		return ret;
	}

	public ResultBox[] detect_face(INDArray img) {
		double scales[] = getScales(img, minSize, factor);
        log.debug("scales loaded, scales are {}.", Arrays.toString(scales));
		INDArray totalBoxes = null;
		for (double scale : scales) {
			INDArray pnetInput = transposeBorder(scaleAndNorm(img, scale));
			INDArray[] out = pnet.output(pnetInput);
			INDArray score = out[0].get(point(0), point(0), all(), all()), reg = out[1];
			INDArray boxes = generateBox(score, reg, thresholds[0], scale);
			if (boxes == null) {
				continue;
			}
			boxes = nms(boxes, 0.5, false);
			totalBoxes = mergeBoxes(totalBoxes, boxes);
		}
        log.debug("pnet scan finished.");
		if (totalBoxes == null) {
			return null;
		}
		totalBoxes = nms(totalBoxes, 0.7, false);
		totalBoxes = bbreg(totalBoxes);
		long[] imgShape = shape(img);
		totalBoxes = rerec(totalBoxes, imgShape[3], imgShape[2]);
		INDArray rnetInput = transposeBorder(reshapeAndNorm(img, totalBoxes, 24));
		INDArray rnetOut[] = rnet.output(rnetInput);
		INDArray score = rnetOut[0].get(all(), point(1)).transposei(), reg = rnetOut[1];
		INDArray ipass = findFitIndexes(score, ">", thresholds[1]);
		if (ipass == null) {
			return null;
		}
		totalBoxes = clearOneDim(get(mergeRegAndScore(totalBoxes, reg, score), ipass).dup());
		totalBoxes = nms(totalBoxes, 0.7, false);
		totalBoxes = bbreg(totalBoxes);
		totalBoxes = rerec(totalBoxes, imgShape[3], imgShape[2]);
		INDArray onetInput = transposeBorder(reshapeAndNorm(img, totalBoxes, 48));
		INDArray onetOut[] = onet.output(onetInput);
		score = onetOut[0].get(all(), point(1)).transposei();
		reg = onetOut[1];
		ipass = findFitIndexes(score, ">", thresholds[2]);
		if (ipass == null) {
			return null;
		}
		totalBoxes = clearOneDim(get(mergeRegAndScore(totalBoxes, reg, score), ipass).dup());
		totalBoxes = bbreg(totalBoxes);
		totalBoxes = nms(totalBoxes, 0.7, true);
		if (totalBoxes == null) {
			return null;
		}
		if (totalBoxes.rank() == 1) {
			totalBoxes = Nd4j.expandDims(totalBoxes, 0);
		}
		ResultBox[] boxes = new ResultBox[(int) totalBoxes.shape()[0]];
		for (int i = 0; i < boxes.length; i++) {
			boxes[i] = new ResultBox(totalBoxes.getInt(i, 0), totalBoxes.getInt(i, 1), totalBoxes.getInt(i, 2),
					totalBoxes.getInt(i, 3));
		}
		return boxes;
	}

	private INDArray mergeRegAndScore(INDArray totalBoxes, INDArray reg, INDArray score) {
		totalBoxes = mergeReg(totalBoxes, reg);
		totalBoxes.put(new INDArrayIndex[] { all(), point(4) }, score.dup().transposei());
		return totalBoxes;
	}

	private INDArray mergeReg(INDArray totalBoxes, INDArray reg) {
		totalBoxes.put(new INDArrayIndex[] { all(), interval(5, 9) }, reg);
		return totalBoxes;
	}

	private INDArray reshapeAndNorm(INDArray img, INDArray totalBoxes, int border) {
		long[] boxShape = shape(totalBoxes);
		INDArray ret = Nd4j.create(boxShape[0], img.shape()[1], border, border);
		for (int i = 0; i < boxShape[0]; i++) {
			INDArray reshapedImg = imresample(
					img.get(all(), all(), interval(totalBoxes.getInt(i, 1), totalBoxes.getInt(i, 3)),
							interval(totalBoxes.getInt(i, 0), totalBoxes.getInt(i, 2))).dup(),
					border, border);
			ret.put(new INDArrayIndex[] { point(i), all(), all(), all() }, reshapedImg);
		}
		return ret.subi(127.5).muli(0.0078125);
	}

	private INDArray rerec(INDArray totalBoxes, double imgW, double imgH) {
		INDArray x1 = totalBoxes.get(all(), point(0)).dup();
		INDArray y1 = totalBoxes.get(all(), point(1)).dup();
		INDArray x2 = totalBoxes.get(all(), point(2)).dup();
		INDArray y2 = totalBoxes.get(all(), point(3)).dup();
		INDArray w = x2.sub(x1);
		INDArray h = y2.sub(y1);
		INDArray regl = maximum(w, h.dup());
		INDArray lossW = regl.sub(w).muli(0.5);
		INDArray lossH = regl.sub(h).muli(0.5);
		totalBoxes.put(new INDArrayIndex[] { all(), point(0) }, fix(maximum(0.0, x1.subi(lossW))));
		totalBoxes.put(new INDArrayIndex[] { all(), point(1) }, fix(maximum(0.0, y1.subi(lossH))));
		totalBoxes.put(new INDArrayIndex[] { all(), point(2) }, fix(minimum(imgW, x2.addi(lossW))));
		totalBoxes.put(new INDArrayIndex[] { all(), point(3) }, fix(minimum(imgH, y2.addi(lossH))));
		return totalBoxes;
	}

	private INDArray bbreg(INDArray totalBoxes) {
		INDArray x1 = totalBoxes.get(all(), point(0)).dup();
		INDArray y1 = totalBoxes.get(all(), point(1)).dup();
		INDArray x2 = totalBoxes.get(all(), point(2)).dup();
		INDArray y2 = totalBoxes.get(all(), point(3)).dup();
		INDArray regw = x2.sub(x1);
		INDArray regh = y2.sub(y1);
		totalBoxes.put(new INDArrayIndex[] { all(), point(0) },
				x1.addi(totalBoxes.get(all(), point(5)).dup().muli(regw)));
		totalBoxes.put(new INDArrayIndex[] { all(), point(1) },
				y1.addi(totalBoxes.get(all(), point(6)).dup().muli(regh)));
		totalBoxes.put(new INDArrayIndex[] { all(), point(2) },
				x2.addi(totalBoxes.get(all(), point(7)).dup().muli(regw)));
		totalBoxes.put(new INDArrayIndex[] { all(), point(3) },
				y2.addi(totalBoxes.get(all(), point(8)).dup().muli(regh)));
		return totalBoxes;
	}

	private INDArray mergeBoxes(INDArray totalBoxes, INDArray boxes) {
		if(boxes.rank() == 1) {
			boxes = Nd4j.expandDims(boxes, 0);
		}
		if (totalBoxes == null) {
			totalBoxes = boxes;
		} else {
			return Nd4j.concat(0, totalBoxes, boxes);
		}
		return totalBoxes;
	}

	private ArrayList<Integer> tmpSelectedIndex = new ArrayList<Integer>();

	private INDArray nms(INDArray boxes, double t, boolean isMethodMin) {
		tmpSelectedIndex.clear();
		INDArray boxTmp = boxes.dup().transposei();
		INDArray x1 = boxTmp.get(point(0), all()).dup().reshape(boxes.shape()[0]);
		INDArray y1 = boxTmp.get(point(1), all()).dup().reshape(boxes.shape()[0]);
		INDArray x2 = boxTmp.get(point(2), all()).dup().reshape(boxes.shape()[0]);
		INDArray y2 = boxTmp.get(point(3), all()).dup().reshape(boxes.shape()[0]);
		INDArray area = x2.sub(x1).addi(1).muli(y2.sub(y1).addi(1));
		INDArray sortedScoreIndex = getScoreSortedIndex(boxes);
		while (true) {
			int lastIndex = (int) sortedScoreIndex.length() - 1;
			int highestIndex = sortedScoreIndex.getInt(lastIndex);
			tmpSelectedIndex.add(highestIndex);
			if (lastIndex == 0) {
				break;
			}
			INDArray otherIndexes = sortedScoreIndex.get(interval(0, lastIndex));
			INDArray xx1 = maximum(x1.getDouble(highestIndex), get(x1, otherIndexes));
			INDArray yy1 = maximum(y1.getDouble(highestIndex), get(y1, otherIndexes));
			INDArray xx2 = minimum(x2.getDouble(highestIndex), get(x2, otherIndexes));
			INDArray yy2 = minimum(y2.getDouble(highestIndex), get(y2, otherIndexes));
			INDArray w = maximum(0.0, xx2.sub(xx1).addi(1));
			INDArray h = maximum(0.0, yy2.sub(yy1).addi(1));
			INDArray interArea = w.mul(h);
			INDArray o;
			if (isMethodMin) {
				o = interArea.divi(minimum(area.getDouble(highestIndex), get(area, otherIndexes)));
			} else {
				o = interArea.divi(get(area, otherIndexes).addi(area.getDouble(highestIndex)).subi(interArea));
			}
			INDArray tmpFitIdx = findFitIndexes(o, "<=", t);
			if (tmpFitIdx == null) {
				break;
			}
			sortedScoreIndex = get(sortedScoreIndex, tmpFitIdx);
		}
		INDArray retIdx = Nd4j.create(new int[] { tmpSelectedIndex.size() });
		for (int i = 0; i < tmpSelectedIndex.size(); i++) {
			retIdx.putScalar(i, tmpSelectedIndex.get(i));
		}
		INDArray ret = get(boxes, retIdx);
		ret = clearOneDim(ret);
		if(ret.rank() == 1) {
			ret = Nd4j.expandDims(ret, 0);
		}
		assert ret.shape()[1] == 9;
		return clearOneDim(ret);
	}

	private INDArray clearOneDim(INDArray src) {
		long[] shape = src.shape();
		int len = shape.length;
		for (long s : shape) {
			if (s == 1) {
				len--;
			}
		}
		if (len == shape.length) {
			return src;
		}
		long[] ret = new long[len];
		int retIdx = 0;
		for (long s : shape) {
			if (s != 1) {
				ret[retIdx] = s;
				retIdx++;
			}
		}
		return src.reshape(ret);
	}

	private INDArray getScoreSortedIndex(INDArray boxes) {
		int scoreIndex = 4;
		long boxShape[] = boxes.shape();
		ArgSortParam arr[] = new ArgSortParam[(int) boxShape[0]];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = new ArgSortParam(boxes.getDouble(i, scoreIndex), i);
		}
		Arrays.sort(arr);
		INDArray ret = Nd4j.create(arr.length);
		for (int i = 0; i < arr.length; i++) {
			ret.putScalar(i, arr[i].index);
		}
		return ret;
	}

	private class ArgSortParam implements Comparable<ArgSortParam> {
		double value;
		int index;

		public ArgSortParam(double value, int index) {
			super();
			this.value = value;
			this.index = index;
		}

		public int compareTo(ArgSortParam obj) {
			double val = this.value - obj.value;
			if (val > 0)
				return 1;
			if (val < 0)
				return -1;
			return 0;
		}
	}

	private INDArray maximum(final Object obj, final INDArray arr) {
		iterateINDArray(arr, new INDArrayIterator() {
			public void iterate(long[] index, double tmp) {
				double d;
				if (obj instanceof INDArray) {
					d = ((INDArray) obj).getDouble(index);
				} else {
					d = ((Number) obj).doubleValue();
				}
				if (d > tmp) {
					arr.putScalar(index, d);
				}
			}
		});
		return arr;
	}

	private INDArray minimum(final Object obj, final INDArray arr) {
		iterateINDArray(arr, new INDArrayIterator() {
			public void iterate(long[] index, double tmp) {
				double d;
				if (obj instanceof INDArray) {
					d = ((INDArray) obj).getDouble(index);
				} else {
					d = ((Number) obj).doubleValue();
				}
				if (d < tmp) {
					arr.putScalar(index, d);
				}
			}
		});
		return arr;
	}

	private INDArray generateBox(INDArray score, INDArray reg, double threshold, double scale) {
		assert reg.rank() == 4 && reg.shape()[0] == 1;
		int stride = 2, cellSize = 12;
		INDArray fitIndexes = findFitIndexes(score, ">=", threshold);
		if (fitIndexes == null) {
			return null;
		}
		INDArray upperLeft = fix(fitIndexes.mul(stride).addi(1).divi(scale).transposei()),
				bottomRight = fix(fitIndexes.mul(stride).addi(cellSize).divi(scale).transpose());
		return Nd4j.hstack(upperLeft, bottomRight, get(score, fitIndexes).transposei(),
				get(reg.get(point(0), point(3), all(), all()), fitIndexes).transposei(),
				get(reg.get(point(0), point(2), all(), all()), fitIndexes).transposei(),
				get(reg.get(point(0), point(1), all(), all()), fitIndexes).transposei(),
				get(reg.get(point(0), point(0), all(), all()), fitIndexes).transposei());
	}

	private ArrayList<double[]> tmp = new ArrayList<double[]>();

	private INDArray findFitIndexes(INDArray arr, final String op, final double t) {
		tmp.clear();
		long shape[] = shape(arr);
		iterateINDArray(arr, new INDArrayIterator() {
			public void iterate(long[] index, double value) {
				if (getByOpType(value, op, t)) {
					double[] copy = new double[index.length];
					for (int i = 0; i < copy.length; i++) {
						copy[i] = index[i];
					}
					tmp.add(copy);
				}
			}
		});
		if (tmp.isEmpty()) {
			return null;
		}
		final INDArray ret = Nd4j.create(shape.length, tmp.size());
		for (int i = 0; i < tmp.size(); i++) {
			ret.put(new INDArrayIndex[] { all(), point(i) }, Nd4j.create(tmp.get(i)).transposei());
		}
		return ret;
	}

	private void iterateINDArray(INDArray arr, INDArrayIterator iter) {
		long shape[] = shape(arr);
		doIterateINDArray(arr, shape, new long[shape.length], 0, iter);
	}

	private void doIterateINDArray(INDArray arr, long[] shape, long[] index, int begin, INDArrayIterator iter) {
		if (begin == index.length) {
			iter.iterate(index, arr.getDouble(index));
		} else {
			for (long i = 0; i < shape[begin]; i++) {
				index[begin] = i;
				doIterateINDArray(arr, shape, index, begin + 1, iter);
			}
		}
	}

	private boolean getByOpType(double d1, String op, double d2) {
		if (">".equals(op))
			return d1 > d2;
		if ("<".equals(op))
			return d1 < d2;
		if ("=".equals(op))
			return d1 == d2;
		if ("<=".equals(op))
			return d1 <= d2;
		if (">=".equals(op))
			return d1 >= d2;
		throw new IllegalArgumentException("invalid op:" + op);
	}

	private INDArray transposeBorder(INDArray arr) {
		long[] shape = shape(arr);
		INDArray ret = Nd4j.create(new long[] { shape[0], shape[1], shape[3], shape[2] });
		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {
				INDArrayIndex[] idx = new INDArrayIndex[] { point(i), point(j), all(), all() };
				ret.put(idx, arr.get(idx).transposei());
			}
		}
		return ret;
	}

	private INDArray scaleAndNorm(INDArray img, double scale) {
		long[] shape = shape(img);
		INDArray ret = imresample(img, (int) Math.ceil(shape[2] * scale), (int) Math.ceil(shape[3] * scale));
		return ret.subi(127.5).muli(0.0078125);
	}

	private double[] getScales(INDArray img, int minSize, double factor) {
		ArrayList<Double> scales = new ArrayList<Double>();
		long[] imgShape = shape(img);
		assert imgShape.length == 4;
		double m = 12.0 / minSize;
		double minl = Math.min(imgShape[2], imgShape[3]) * m;
		int factorCount = 0;
		while (minl >= 12) {
			scales.add(m * Math.pow(factor, factorCount));
			minl *= factor;
			factorCount++;
		}
		double[] ret = new double[scales.size()];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = scales.get(i).doubleValue();
		}
		return ret;
	}

	private long[] shape(INDArray img) {
		long[] ret = img.shape();
		if (ret.length == 2 && ret[0] == 1) {
			ret = new long[] { ret[1] };
		}
		if (ret.length == 0) {
			ret = new long[] { 1 };
		}
		return ret;
	}

	private ByteArrayOutputStream tmpBos = new ByteArrayOutputStream();
	private INDArray imresample(INDArray img, int hs, int ws) {
		long[] shape = img.shape();
		long h = shape[2];
		long w = shape[3];
		float dx = (float) w / ws;
		float dy = (float) h / hs;
		INDArray im_data = Nd4j.create(new long[] { 1, 3, hs, ws });
		for (int a1 = 0; a1 < 3; a1++) {
			for (int a2 = 0; a2 < hs; a2++) {
				for (int a3 = 0; a3 < ws; a3++) {
					im_data.putScalar(new long[] { 0, a1, a2, a3 },
							img.getDouble(0, a1, (long) Math.floor(a2 * dy), (long) Math.floor(a3 * dx)));
				}
			}
		}
		return im_data;
//		return fromImg(ImageLoader.toBufferedImage(toImage(img)
//				.getScaledInstance(ws, hs, BufferedImage.SCALE_SMOOTH), 
//				BufferedImage.TYPE_INT_RGB));
	}
	
	public static BufferedImage toImage(INDArray imgINDArray) {
		long[] shape = imgINDArray.shape();
		BufferedImage img = new BufferedImage((int) shape[3], (int) shape[2], BufferedImage.TYPE_INT_RGB);
		WritableRaster raster = img.getRaster();
		for (int i = 0; i < img.getWidth(); i++) {
			for (int j = 0; j < img.getHeight(); j++) {
				raster.setPixel(i, j, new double[] { imgINDArray.getDouble(0, 2, j, i),
						imgINDArray.getDouble(0, 1, j, i), imgINDArray.getDouble(0, 0, j, i) });
			}
		}
		return img;
	}
	
	private INDArray fromImg(BufferedImage img) {
		INDArray imgINDArray = Nd4j.create(1,3,img.getHeight(),img.getWidth());
		WritableRaster raster = img.getRaster();
		for (int i = 0; i < img.getWidth(); i++) {
			for (int j = 0; j < img.getHeight(); j++) {
				double[] rgb = raster.getPixel(i, j, new double[3]);
				imgINDArray.put(new INDArrayIndex[] {point(0), all(), point(j), point(i)}, 
						Nd4j.reverse(Nd4j.create(rgb).transpose()));
			}
		}
		return imgINDArray;
	}

	private INDArray fix(INDArray src) {
		long shape[] = src.shape();
		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {
				src.putScalar(i, j, src.getInt(i, j));
			}
		}
		return src;
	}

	private INDArray get(INDArray src, INDArray idxes) {
		long idxShape[] = shape(idxes);
		if(idxes.length() == 1 && idxes.getInt(0) == 0 && src.shape()[0] == 1) {
			long s[] = new long[src.rank() - 1];
			System.arraycopy(src.shape(), 1, s, 0, s.length);
			return src.reshape(s);
		}
		if (idxShape.length == 1 && idxShape[0] == 1 && src.shape().length > idxShape.length) {
			INDArrayIndex[] arrIdx = new INDArrayIndex[src.shape().length];
			for (int i = 0; i < arrIdx.length; i++) {
				arrIdx[i] = all();
			}
			if (src.shape().length == 2 && src.shape()[0] == 1) {
				arrIdx[1] = point(idxes.getInt(0));
			} else {
				arrIdx[0] = point(idxes.getInt(0));
			}
			return src.get(arrIdx);
		}
		return src.get(idxes);
	}
}
