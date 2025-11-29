package org.ppoplib.ppo.nn;

import org.ppoplib.linalg.Matrix;
import org.ppoplib.ppo.act.Activation;
import org.ppoplib.ppo.act.ActivationType;
import org.ppoplib.ppo.act.IActivation;
import java.util.Random;

public class Layer {
    int nodesIn;
    int nodesOut;
    Matrix weights;
    Matrix biases;
    Matrix weightGradients;
    Matrix biasGradients;
    IActivation activation;

    public Layer(int in, int out, IActivation act){
        nodesIn = in;
        nodesOut = out;
        weights = new Matrix(out, in);
        biases = new Matrix(out, 1);
        this.activation = act;
        clearGradients();
    }

    public void randomizeLayer(Random rand){
        if(rand == null){
            rand = new Random();
        }
        for(int r = 1; r <= nodesOut; r++){
            for(int c = 1; c <= nodesIn; c++) {
                weights.setEntry(r,c,rand.nextDouble(-1,1)/Math.sqrt(nodesOut));
            }
            biases.setEntry(r,1,rand.nextDouble(-1,1)/Math.sqrt(nodesOut));
        }
    }

    public void updateGradients(LearnData ld){

    }

    public void applyGradients(double learnRate){
        weights.add(weightGradients.scale(-learnRate)); // w -= w_grad * lr
        biases.add(biasGradients.scale(-learnRate));    // b -= b_grad * lr
    }

    public void clearGradients(){
        weightGradients = new Matrix(nodesOut, nodesIn);
        biasGradients = new Matrix(nodesOut, 1);
    }

    public double[] feed(double[] inputs){
        System.out.println("Activation type: "+(activation.getActivationType() == ActivationType.LINEAR ? "linear" : "tanh"));
        Matrix inputMatrix = new Matrix(inputs);
        double[] preactivatedOutputs = weights.multiply(inputMatrix).add(biases).toArray();
        double[] outputs = new double[nodesOut];
        for(int i = 0; i < preactivatedOutputs.length; i++){
            outputs[i] = activation.activate(preactivatedOutputs, i);
        }
        return outputs;
    }

    public String toString(){
        String s = "";
        s+=String.format("%d in, %d out\n", nodesIn,nodesOut);
        s+="Weights:\n"+weights.toString();
        s+="Biases:\n"+biases.toString();
        return s;
    }
}