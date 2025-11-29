package org.ppoplib.ppo.env;

import org.ppoplib.ppo.nn.LearnData;

public class Observation extends LearnData {

    public Observation(double[] inputs, double[] expectedOutputs) {
        super(inputs, expectedOutputs);
    }

}
