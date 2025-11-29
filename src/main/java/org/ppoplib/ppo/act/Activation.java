package org.ppoplib.ppo.act;

import org.ppoplib.linalg.Matrix;

public class Activation {

    public static IActivation getActivation(ActivationType type){
        return switch (type) {
            case SIGMOID -> new Sigmoid();
            case TANH -> new Tanh();
            case RELU -> new Relu();
            case SOFTMAX -> new Softmax();
            case LINEAR -> new Linear();
            default -> {
                System.err.println("Unimplemented activation type");
                yield new Sigmoid();
            }
        };
    }

    public static class Sigmoid implements IActivation{
        @Override
        public double activate(double[] inputs, int index) {
            return 1./(1. + Math.exp(-inputs[index]));
        }

        @Override
        public double derivative(double[] inputs, int index) {
            double theta = activate(inputs, index);
            return theta * (1-theta);
        }

        @Override
        public ActivationType getActivationType() {
            return ActivationType.SIGMOID;
        }
    }

    public static class Tanh implements IActivation{
        @Override
        public double activate(double[] inputs, int index) {
            return Math.tanh(inputs[index]);
        }

        @Override
        public double derivative(double[] inputs, int index) {
            double tanh = activate(inputs, index);
            return 1 - tanh * tanh;
        }

        @Override
        public ActivationType getActivationType() {
            return ActivationType.TANH;
        }
    }

    public static class Relu implements IActivation{
        @Override
        public double activate(double[] inputs, int index) {
            double d = inputs[index];
            return d > 0 ? d : 0;
        }

        @Override
        public double derivative(double[] inputs, int index) {
            return inputs[index] > 0 ? 1 : 0;
        }

        @Override
        public ActivationType getActivationType() {
            return ActivationType.RELU;
        }
    }

    public static class Softmax implements IActivation{
        @Override
        public double activate(double[] inputs, int index) {
            double sum = 0;
            for(double input : inputs){
                sum += Math.exp(input);
            }
            return Math.exp(inputs[index]) / sum;
        }

        @Override
        public double derivative(double[] inputs, int index) {
            double sum = 0;
            for(double input : inputs){
                sum += Math.exp(input);
            }
            double exp = Math.exp(inputs[index]);
            return (exp * sum - exp * exp) / (sum * sum);
        }

        @Override
        public ActivationType getActivationType() {
            return ActivationType.SOFTMAX;
        }
    }

    public static class Linear implements IActivation{
        @Override
        public double activate(double[] inputs, int index) {
            return inputs[index];
        }

        @Override
        public double derivative(double[] inputs, int index) {
            return 1;
        }

        @Override
        public ActivationType getActivationType() {
            return ActivationType.LINEAR;
        }
    }
}
