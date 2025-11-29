package org.ppoplib.linalg;

public class Matrix {
    private int r;
    private int c;
    private double[] data;

    public Matrix(int rows, int cols){
        if(rows < 1 || cols < 1){
            throw new IllegalArgumentException("Matrix of size "+rows+"x"+cols+" does not exist");
        }
        this.r = rows;
        this.c = cols;
        data = new double[r * c];
    }
    public Matrix(double[] column){
        if(column == null || column.length == 0) {
            throw new IllegalArgumentException("No data given for column matrix");
        }
        this.r = column.length;
        this.c = 1;
        data = column;
    }

    public double getEntry(int row, int col){
        if(row < 1 || row > r || col < 1 || col > c){
            throw new IllegalArgumentException("Requesting a matrix entry that does not exist");
        } else return data[(row-1) * c + (col-1)];
    }

    public double setEntry(int row, int col, double e){
        if(row < 1 || row > r || col < 1 || col > c){
            throw new IllegalArgumentException("Requesting a matrix entry that does not exist");
        }
        double temp = getEntry(row, col);
        data[(row-1) * c + (col-1)] = e;
        return temp;
    }

    public int getRows(){
        return r;
    }
    public int getCols(){
        return c;
    }

    public Matrix getRow(int row){
        if(row < 1 || row > c) throw new IllegalArgumentException("Invalid row in "+r+"x"+c+" matrix");
        Matrix A_row = new Matrix(r,1);
        for(int col = 1; col <= r; col++){
            A_row.setEntry(1,col,this.getEntry(row,col));
        }
        return A_row;
    }
    public Matrix getColumn(int col){
        if(col < 1 || col > c) throw new IllegalArgumentException("Invalid column in "+r+"x"+c+" matrix");
        Matrix A_col = new Matrix(r,1);
        for(int row = 1; row <= r; row++){
            A_col.setEntry(row,1,this.getEntry(row,col));
        }
        return A_col;
    }

    public double[] toArray(){
        double[] data2 = new double[data.length];
        for(int i = 0; i < data.length; i++){
            data2[i] = data[i];
        }
        return data2;
    }

    public Matrix add(Matrix other){
        if(r != other.getRows() || c != other.getCols()){
            throw new IllegalArgumentException();
        } else{
            Matrix m = new Matrix(r, c);
            for(int i = 0; i < data.length; i++){
                int row = i / c + 1;
                int col = i % c + 1;
                m.setEntry(row, col, this.getEntry(row, col) + other.getEntry(row, col));
            }
            return m;
        }
    }

    public Matrix scale(double d){
        Matrix m = new Matrix(r, c);
        for(int i = 0; i < data.length; i++){
            int row = i / c + 1;
            int col = i % c + 1;
            m.setEntry(row, col, d * this.getEntry(row, col));
        }
        return m;
    }

    public Matrix multiply(Matrix other){
        if(c != other.getRows()){
            throw new IllegalArgumentException("Impossible to multiply matrix");
        } else {
            Matrix m = new Matrix(r, other.getCols());
            for(int row = 1; row <= r; row++){
                for(int col = 1; col <= other.getCols(); col++){
                    double entry = 0;
                    for(int i = 1; i <= c; i++){
                        entry += this.getEntry(row, i) * other.getEntry(i, col);
                    }
                    m.setEntry(row, col, entry);
                }
            }
            return m;
        }
    }

    public static Matrix getIdentity(int n){
        if(n < 1) throw new IllegalArgumentException("Identity matrix of size < 1 does not exist");
        Matrix I = new Matrix(n, n);
        for(int i = 1; i <= n; i++){
            I.setEntry(i,i,1);
        }
        return I;
    }

    public boolean isSquare(){
        return r == c;
    }

    public boolean hasInverse(){
        try{
            return this.isSquare() && this.det() != 0;
        } catch (UndefinedDeterminantException e){
            return false;
        }
    }

    public Matrix getMinor(int dRow, int dCol){
        if(!this.isSquare()) throw new IllegalStateException("Nonsquare matrix does not have a minor");
        if(dRow > r || dRow < 1 || dCol > c || dCol < 1) throw new IllegalArgumentException("Minor does not exist");
        Matrix minor = new Matrix(r-1,c-1);
        for(int row = 1; row <= r; row++){
            for(int col = 1; col <= c; col++){
                int destRow = row > dRow ? row - 1 : row;
                int destCol = col > dCol ? col - 1 : col;
                if(row != dRow && col != dCol)
                    minor.setEntry(destRow,destCol,this.getEntry(row,col));
            }
        }
        return minor;
    }

    // TODO
    public Matrix getTranspose(){
        return null;
    }

    // TODO
    public Matrix getCofactor(){
        return null;
    }

    public Matrix getInverse() throws UndefinedDeterminantException{
        double det = this.det();
        if(det == 0) return null;
        else return this.getCofactor().scale(1./det);
    }

    public double det(double k) throws UndefinedDeterminantException {
        if(!this.isSquare()) throw new UndefinedDeterminantException();
        if(r==1 && c==1){
            return k * this.getEntry(1,1);
        }
        double det = 0;
        int row = 1;
        for(int col = 1; col <= c; col++){
            det += col%2==1
                    ? this.getEntry(row,col) * getMinor(row,col).det(k)
                    : this.getEntry(row,col) * -1 * getMinor(row,col).det(k);
        }
        return det;
    }
    public double det() throws UndefinedDeterminantException {
        return det(1);
    }

    public String toString(){
        String s = "";
        for(int row = 1; row <= this.r; row++){
            s += "| ";
            for(int col = 1; col <= this.c; col++){
                s += this.getEntry(row,col)+" ";
            }
            s += "|\n";
        }
        return s;
    }
}
