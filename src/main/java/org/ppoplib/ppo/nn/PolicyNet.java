package org.ppoplib.ppo.nn;

import org.ppoplib.ppo.act.Activation;
import org.ppoplib.ppo.act.ActivationType;
import org.ppoplib.ppo.act.IActivation;

import java.util.Random;

public class PolicyNet extends NeuralNet {
    /** A list of independent action spaces. Each index i represents the i-th discrete action space,
     * where dSpaces[i] is the number of possible outputs */
    private final int[] dSpaces;
    /** A list of independent action spaces. Each index i represents the i-th continuous action space,
     * where the possible outputs are in the range from [0, cSpaces[i]] */
    private final double[] cSpaces;
    /** When training, randomness is injected into the agent's actions. Thus, the output of the neural network,
     * which is stored in this array, is used to parameterize a random distribution. The final outputs of the
     * network in this case are then sampled from this random distribution. */
    private double[] parameterization;
    /** A list of probabilities representing the probability that each independent action would have been
     * sampled from the current parameterization. Here, pDist.length <= parameterization.length */
    private double[] pDist;
    
    private static final IActivation softmax = new Activation.Softmax();
    private static final double stDev = 0.5;

    
    public PolicyNet(int numInputs, int[] hiddenLayerSizes, int[] discreteSpaces, double[] continuousSpaces) {
        super(numInputs, sum(discreteSpaces) + continuousSpaces.length, hiddenLayerSizes, ActivationType.SIGMOID);
        this.dSpaces = discreteSpaces;
        this.cSpaces = continuousSpaces;
        this.parameterization = null;
        this.pDist = null;
    }

    /**
     * Samples a random action based on a given environment state.
     * More specifically, the state is first fed through the neural network.
     * For each discrete action space, softmax is applied to all output nodes in the space to create a
     * probability table for the space.
     * For each continuous action space, the network output (0,1) is scaled to (0, cSpace[i]), where
     * i is the index of the continuous action space.
     * Finally, if the injectRandomness flag is set,
     * for each discrete space: an index is selected randomly based on the probability table, and
     * for each continuous space: a double is selected from a gaussian distribution with the network output
     * as the mean and a scalar of the hyperparameter stdev as the standard deviation.
     * @param state the double[] representation of the current state
     * @param rand the random number generator to use, or null to create a new one
     * @param injectRandomness slightly randomizes the network's output if true, returns the deterministic
     *                         output if false
     * @return an array of doubles representing the sampled action
     */
    public double[] sampleAction(double[] state, Random rand, boolean injectRandomness) {
        if(rand == null) rand = new Random();
        parameterization = feed(state);
        double[] sampledAction = new double[cSpaces.length + dSpaces.length];
        // apply softmax to groups of output nodes representing discrete actions, then sample each action
        int currIndex = 0;
        for(int i = 0; i < dSpaces.length; i++){
            double[] currSpace = new double[dSpaces[i]];
            System.arraycopy(parameterization, currIndex, currSpace, 0, dSpaces[i]);
            for(int j = 0; j < dSpaces[i]; j++){
                parameterization[i] = softmax.activate(currSpace, i);
            }
            sampledAction[i] = injectRandomness
                    ? sampleFromTable(currSpace, rand)
                    : indexOfMax(currSpace);
            currIndex += dSpaces[i];
        }
        // scale output nodes representing continuous actions to fill the space, then sample each action
        for(int i = parameterization.length - 1; i >= parameterization.length - cSpaces.length; i--){
            parameterization[i] *= cSpaces[i];
            if(injectRandomness){
                double mean = parameterization[i];
                double stdev = stDev / cSpaces[i];
                sampledAction[i] = Math.clamp(rand.nextGaussian(mean, stdev), 0, cSpaces[i]);
            } else {
                sampledAction[i] = parameterization[i];
            }
        }
        return sampledAction;
    }

    public double[] getParameterization(){
        if(parameterization == null) throw new IllegalStateException("Need to run inputs through the network first");
        return parameterization;
    }

    /**
     *
     * @return
     */
    public double[] getProbabilities(){
        // TODO
        return null;
    }

    @Override
    protected void updateAllGradients(LearnData ld){

    }

    private static int sampleFromTable(double[] dist, Random rand){
        double r = rand.nextDouble(0,1);
        double sum = 0;
        for(int i = 0; i < dist.length; i++){
            sum += dist[i];
            if(r < sum) return i;
        }
        return 0;
    }

    private static int indexOfMax(double[] a){
        int maxIndex = 0;
        for(int i = 0; i < a.length; i++){
            if(a[i] > a[maxIndex]){
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private static int sum(int[] a){
        int sum = 0;
        for(int e : a){ sum += e; }
        return sum;
    }
}
