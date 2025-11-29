package org.ppoplib.ppo.nn;

import org.ppoplib.ppo.act.ActivationType;

public class StateValueNet extends NeuralNet {
    public StateValueNet(int[] layerSizes) {
        super(layerSizes, ActivationType.LINEAR, ActivationType.LINEAR);
    }
}
