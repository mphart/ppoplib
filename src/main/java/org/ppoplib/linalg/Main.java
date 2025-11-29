package org.ppoplib.linalg;

import java.util.Random;

public class Main {
    public static void main(String[] args){
        Random rand = new Random();

        Matrix m = new Matrix(3,3);
        Matrix m2 = new Matrix(3,1);
        Matrix m3 = new Matrix(14, 14);
        Matrix m4 = new Matrix(14, 14);

        for(int r = 1; r <= m.getRows(); r++){
            for(int c = 1; c <= m.getCols(); c++){
                m.setEntry(r,c,rand.nextInt(-10,11));
            }
        }
        for(int r = 1; r <= m2.getRows(); r++){
            for(int c = 1; c <= m2.getCols(); c++){
                m2.setEntry(r,c,rand.nextInt(-10,11));
            }
        }
        for(int r = 1; r <= m3.getRows(); r++){
            for(int c = 1; c <= m3.getCols(); c++){
                m3.setEntry(r,c,rand.nextInt(-10,11));
                m4.setEntry(r,c,rand.nextInt(-10,11));
            }
        }

        System.out.println(m);
        System.out.println(m2);
        System.out.println(m.multiply(m2));
        System.out.println(m3);
        System.out.println(m4);
        System.out.println(m3.add(m4));
        System.out.println(m3.add(m4).scale(-2));

        System.out.println(m.getMinor(1,1));
        System.out.println(m.getMinor(1,2));
        System.out.println(m.getMinor(2,1));
        System.out.println(m.getMinor(2,2));

        System.out.println(Matrix.getIdentity(3));
        System.out.println(Matrix.getIdentity(3).multiply(m));
        try{
            System.out.println(m);
            System.out.println(m.det());
            System.out.println(m.det(2));
        } catch (UndefinedDeterminantException e){}
    }
}
