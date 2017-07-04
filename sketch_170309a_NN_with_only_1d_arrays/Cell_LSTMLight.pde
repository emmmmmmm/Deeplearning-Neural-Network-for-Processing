//================================
// a simplified LSTM cell ("mix-gate-cell")
//
//================================
class MixGateCell extends  BaseCell{ //  -> maybe do that with layers instead?^^ ... idk...
  float[] x,dx;
  float Sx, Sh;
  float G, dG;
  float dSx, dSh;
  float[] Wm, dWm, Wx, dWx;
  float B, dB;
  float Bx, dBx;
  float ShPrev;
  int numInputs;
  float learningRate = 1e-3;

  //================================
  MixGateCell(int num, int t) {
    numInputs = num;
    initArrays();
  }
  //================================
  public void initArrays() {
    x  = new float[numInputs];
    dx = new float[numInputs];
    Wm  = new float[2];
    dWm = new float[2];
    Wx = new float[numInputs];
    dWx= new float[numInputs];
    for (int i=0; i<Wx.length; i++) Wx[i] = initWeight(2);  //randomize Weights for Inputs

    Wm[1] = 0.21f;  // hard init Weights for Mix-Gate
    Wm[0] = 1.2f;
  }
  //================================
  public float forward(float[] _x) {
    x = _x;
    ShPrev = Sh;
    Sx = 0;
    for (int in=0; in<numInputs; in++)
      Sx += Wx[in] * x[in];
    Sx += Bx;
    G = softSign( Sx * Wm[0] + ShPrev * Wm[1] + B); // "mix-gate"
    Sh = Sx * G + ShPrev * (- G);      // mix Sx and Sh[t-1]
    Sh = softSign(Sh);

    return Sh;
  }
  //================================
  public void resetStates() {
    Sh = 0;
    Sx = 0;
  }
  //================================
  // calculate gradients
  public float[] backward(float error) {

    dSh = error;            // error for this timestep
    dSh = dSoftSign(Sh) * dSh; // softsign

    dG = (Sx-ShPrev) * dSh;
    dSx = G * dSh;

    dG = dSoftSign(G) * dG; // softsign
    dB += dG;
    dWm[0] += Sx * dG;
    dWm[1] += ShPrev*dG;

    dBx += dSx;
    dx = new float[dx.length];
    for (int in=0; in<numInputs; in++) {
      dWx[in] += dSx * x[in];
      dx[in]  += dSx * Wx[in];
    }
    return dx;
  }
  //================================
  // update weights and biases
  public void update(float lr) {
    learningRate = lr;
    for (int i=0; i<Wx.length; i++)
      Wx[i] += dWx[i]*learningRate;
    for (int i=0; i<Wm.length; i++)
      Wm[i] += dWm[i]*learningRate;

    Bx += dBx * learningRate;
    B  += dB  * learningRate;
    resetGradients();

    //generalize:
    if (false) {
      for (int i=0; i<Wx.length; i++)
        Wx[i] *=0.99999f;
      for (int i=0; i<Wm.length; i++)
        Wm[i]*=0.99999f;
    }
  }
  //================================
  private void resetGradients() {
    dWx = new float[dWx.length];
    dWm = new float[dWm.length];
    dB = 0;
    dBx = 0;
  }
  //================================
  // pushes value into array (from top) (bottom-value drops out)
  private void push(float[] ar, float f) {
    for (int i=1; i<ar.length; i++)
      ar[i-1] = ar[i];
    ar[ar.length-1] = f;
  }
  //================================
  private void push(float[][] ar, float[] f) {
    for (int i=1; i<ar.length; i++)
      for (int j=0; j<ar[i].length; j++)
        ar[i-1][j] = ar[i][j];
    for (int j = 0; j < f.length; j++)
      ar[ar.length-1][j] = f[j];
  }
  //================================
  private float sigmoid(float val) {return 1 / (1 + exp(-1 * val));  }
  //================================
  private float softSign(float x) {return 2.0 * x / (1.0 + abs(x));  }
  //================================
  private float dSoftSign(float x) {return 2.0 / sq(1.0 + abs(x));  }
  //================================
  public void setLearningRate(float lr) {learningRate = lr;  }
  //================================
  private float initWeight(float w) {return random(2*w)-w;  }
  //================================
  private float absSum(float[] ar) {
    float ret=0;
    for (int i=0; i<ar.length; i++)
      ret+=abs(ar[i]);
    return ret;
  }
}
