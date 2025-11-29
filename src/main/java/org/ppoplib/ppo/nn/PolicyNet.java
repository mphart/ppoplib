package org.ppoplib.ppo.nn;

import org.ppoplib.ppo.act.ActivationType;

public class PolicyNet extends NeuralNet {
    private int[] dSpaces;
    private double[] cSpaces;
    private double[] probabilityDist;

    public PolicyNet(int[] layerSizes, int[] discreteSpaces, double[] continuousSpaces, ActivationType inAct) {
        // linear output activation since we want to execute multiple activation functions piecewise
        super(layerSizes, inAct, ActivationType.LINEAR);
        // check to make sure the given layerSizes and given discrete/continuous action spaces match
         // TODO
        this.dSpaces = discreteSpaces;
        this.cSpaces = continuousSpaces;
        this.probabilityDist = new double[layerSizes[layerSizes.length - 1]];
    }

    // TODO
    public double[] getAction(double[] state){
        probabilityDist = super.feed(state);
        double[] outputs = new double[dSpaces.length + cSpaces.length];
        return null;
    }

    // TODO
    /**
     * Samples a random action from a given environment state.
     * More specifically, the state is first fed through the neural network.
     * For each discrete action space, softmax is applied to all nodes in the space to create a
     * probability table for the space.
     * Finally,
     * for each discrete space: an index is selected randomly based on the probability table
     * for each continuous space: a double
     * @param state
     * @return
     */
    public double[] sampleAction(double[] state) {
        super.feed(state);
        return null;

    }

    @Override
    protected void updateAllGradients(LearnData ld){

    }
}
