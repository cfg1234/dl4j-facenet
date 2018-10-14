package perf.cfg.dl4jfacenet.model.custom;

import java.util.Arrays;

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
import org.nd4j.linalg.factory.Nd4j;
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
        	INDArray ret = toRank2(in);
        	return toRank4(activation.getActivation(ret, training), in);
        } else {
        	return activation.getActivation(in, training);
        }
    }
    
    private INDArray toRank4(INDArray output, INDArray result) {
    	long[] shape = output.shape(),
    			resultShape = result.shape();
    	for(long i = 0;i < shape[0];i++){
    		for(long j = 0;j < shape[1];j++){
    			long tmp = i;
    			long[] index = new long[resultShape.length];
    			index[0] = tmp / (resultShape[2] * resultShape[3]);
    			tmp %= (resultShape[2] * resultShape[3]);
    			index[1] = j;
    			index[2] = tmp / resultShape[3];
    			index[3] = tmp % resultShape[3];
    			result.putScalar(index, output.getDouble(i,j));
    		}
    	}
		return result;
	}

	private INDArray tmp = null;

    private INDArray toRank2(INDArray in) {
		long[] shape = in.shape();
		long[] newShape = new long[]{in.length() / shape[1], shape[1]};
		if(tmp == null?true:Arrays.equals(tmp.shape(), newShape)){
			tmp = Nd4j.createUninitialized(newShape);
		}
		for(long i1 = 0;i1 < shape[0];i1++){
			for(long i2 = 0;i2 < shape[1];i2++){
				for(long i3 = 0;i3 < shape[2];i3++){
					for(long i4 = 0;i4 < shape[3];i4++){
						tmp.putScalar(new long[]{i1 * shape[2] * shape[3] + i3 * shape[3] + i4, i2}, in.getDouble(i1,i2,i3,i4));
					}
				}
			}
		}
		return tmp;
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
