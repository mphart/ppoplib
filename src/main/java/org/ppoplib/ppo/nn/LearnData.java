package org.ppoplib.ppo.nn;

public class LearnData {
    double[] inputs;
    double[] expectedOutputs;

    public LearnData(double[] inputs, double[] expectedOutputs){
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
    }
}
