package perf.cfg.dl4jfacenet.model.custom;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.PReLUParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationPReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

public class PReLUNormLayer extends BaseLayer<PReLUNorm> {

	private static final long serialVersionUID = 1L;

	public PReLUNormLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public PReLUNormLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public Type type() {
        return Type.NORMALIZATION;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr mgr) {
        assertInputSet(false);
        applyDropOutIfNecessary(training, mgr);

        INDArray in;
        if (training) {
            in = mgr.dup(ArrayType.ACTIVATIONS, input, input.ordering());
        } else {
            in = mgr.leverageTo(ArrayType.ACTIVATIONS, input);
        }

        INDArray alpha = getParam(PReLUParamInitializer.WEIGHT_KEY);
        IActivation activation = new ActivationPReLU(alpha);
        if(in.rank() == 4){
        	return activate(in, activation, training);
        } else {
        	return activation.getActivation(in, training);
        }
    }

	private INDArray activate(INDArray in, IActivation activation, boolean training) {
		long inshape[] = in.shape();
		for(int i = 0;i < inshape[0];i++) {
			for(int j = 0;j < inshape[2];j++) {
				for(int k = 0;k < inshape[3];k++) {
					INDArrayIndex[] idx = new INDArrayIndex[] {
							NDArrayIndex.point(i), NDArrayIndex.all(),
							NDArrayIndex.point(j), NDArrayIndex.point(k)
					};
					in.put(idx, activation.getActivation(in.get(idx).transposei(), training));
				}
			}
		}
		return in;
	}

	@Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        INDArray layerInput = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, input, input.ordering());

        INDArray alpha = getParam(PReLUParamInitializer.WEIGHT_KEY);
        IActivation prelu = new ActivationPReLU(alpha);

        Pair<INDArray, INDArray> deltas = prelu.backprop(layerInput, epsilon);
        INDArray delta = deltas.getFirst();
        INDArray weightGradView = deltas.getSecond();

        delta = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, delta);  //Usually a no-op (except for perhaps identity)
        delta = backpropDropOutIfPresent(delta);
        Gradient ret = new DefaultGradient();
        ret.setGradientFor(PReLUParamInitializer.WEIGHT_KEY, weightGradView, 'c');

        return new Pair<Gradient, INDArray>(ret, delta);
    }


    public boolean isPretrainLayer() {
        return false;
    }

}
