package org.ppoplib.ppo;

import java.util.Arrays;
import java.util.Random;
import org.ppoplib.ppo.act.*;
import org.ppoplib.ppo.nn.NeuralNet;
import org.ppoplib.ppo.nn.PolicyNet;

public class NeuralNetTest {
    public static void main(String[] args){

        NeuralNet nn = new NeuralNet(5, 3, new int[]{8,8,8,8}, ActivationType.LINEAR);
        Random rand = new Random(0);

        nn.randomInit(rand);

        System.out.println(nn);
        System.out.println(nn.feed(new double[]{1,2,3,4,5}));

        for(int i = 0; i < 1000; i++){
            System.out.println(rand.nextDouble(0,1));
        }

    }
}
