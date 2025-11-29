package org.ppoplib.ppo.nn;

import org.ppoplib.ppo.act.Activation;
import org.ppoplib.ppo.act.ActivationType;
import org.ppoplib.ppo.act.IActivation;
import java.util.Random;

public class NeuralNet {
    protected int[] layerSizes;
    protected int numInputs;
    protected int numOutputs;
    protected IActivation activation = new Activation.Sigmoid();
    protected Layer[] layers;

    public NeuralNet(int inputNodes, int outputNodes, int[] hiddenLayerSizes, ActivationType actType){
        layerSizes = new int[hiddenLayerSizes.length + 2];
        layerSizes[0] = inputNodes;
        System.arraycopy(hiddenLayerSizes, 0, layerSizes, 1, hiddenLayerSizes.length);
        layerSizes[layerSizes.length - 1] = outputNodes;
        this.numInputs = inputNodes;
        this.numOutputs = outputNodes;
        this.activation = Activation.getActivation(actType);
        for(int i = 0; i < layerSizes.length - 1; i++){
            layers[i] = new Layer(layerSizes[i], layerSizes[i+1], this.activation);
        }
    }

    public void randomInit(Random rand){
        for(int i = 0; i < layers.length; i++){
            layers[i].randomizeLayer(rand);
        }
    }
    public void randomInit(){
        this.randomInit((Random)null);
    }

    public double[] feed(double[] inputs){
        if(inputs == null || inputs.length < numInputs){
            double[] refinedInputs = new double[numInputs];
            if(inputs != null){
                for(int i = 0; i < numInputs; i++){
                    refinedInputs[i] = i < inputs.length ? inputs[i] : 0;
                }
            }
            inputs = refinedInputs;
        }
        if(inputs.length > numInputs) throw new IllegalArgumentException("Invalid number of inputs\n");
        double[] outputs = null;
        for(int i = 0; i < layers.length; i++){
            outputs = layers[i].feed(inputs);
            inputs = outputs;
        }
        return outputs;
    }

    public void learn(LearnData[] trainingData, double learnRate){
        for(LearnData ld : trainingData){
            updateAllGradients(ld);
        }
        applyAllGradients(learnRate / trainingData.length);
        clearAllGradients();
    }

    protected void updateAllGradients(LearnData ld){
        for(Layer l : layers){
            l.updateGradients(ld);
        }
    }

    protected void applyAllGradients(double learnRate){
        for(Layer l : layers){
            l.applyGradients(learnRate);
        }
    }

    protected void clearAllGradients(){
        for(Layer l : layers){
            l.clearGradients();
        }
    }

    public String toString(){
        String s = "";
        for(int i = 0; i < layers.length; i++){
            s+=String.format("Layer %d:\n", i);
            s+=layers[i].toString();
            s+="\n";
        }
        return s;
    }
}
