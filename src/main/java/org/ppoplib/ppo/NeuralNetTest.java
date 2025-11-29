package org.ppoplib.ppo;

import java.util.Arrays;
import java.util.Random;
import org.ppoplib.ppo.act.*;
import org.ppoplib.ppo.nn.NeuralNet;
import org.ppoplib.ppo.nn.PolicyNet;

public class NeuralNetTest {
    public static void main(String[] args){
        NeuralNet nn = new NeuralNet(
                new int[]{16,128,256,256,256,256,12},
                ActivationType.LINEAR,
                ActivationType.TANH
        );
        Random random = new Random(0);
        nn.randomInit(random);
        double[] outputs = nn.feed(new double[]{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1});
        System.out.println(Arrays.toString(outputs));

        IActivation sigmoid = Activation.getActivation(ActivationType.SIGMOID);
        IActivation tanh = Activation.getActivation(ActivationType.TANH);
        IActivation relu = Activation.getActivation(ActivationType.RELU);
        IActivation softmax = Activation.getActivation(ActivationType.SOFTMAX);

        System.out.println("Sigm: "+sigmoid.activate(outputs, 0));
        System.out.println("Tanh: "+tanh.activate(outputs, 0));
        System.out.println("Relu: "+relu.activate(outputs, 0));
        System.out.println("Soft: "+softmax.activate(outputs, 0));

        PolicyNet pn = new PolicyNet(
                new int[]{16,32,32,32,32,16},
                new int[]{2,2,2,5},
                new double[]{2.0, Math.PI, 3, 34, 6.6},
                ActivationType.RELU
        );
    }
}
