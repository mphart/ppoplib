package org.ppoplib.ppo.act;

import org.ppoplib.linalg.Matrix;

public interface IActivation {
    public double activate(double[] inputs, int index);

    public double derivative(double[] inputs, int index);

    public ActivationType getActivationType();
}
