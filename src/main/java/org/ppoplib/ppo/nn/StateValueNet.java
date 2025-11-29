package org.ppoplib.ppo.nn;

import org.ppoplib.ppo.act.ActivationType;

public class StateValueNet extends NeuralNet {

    public StateValueNet(int inputNodes, int outputNodes, int[] hiddenLayerSizes, ActivationType actType) {
        super(inputNodes, outputNodes, hiddenLayerSizes, actType);
    }
}
