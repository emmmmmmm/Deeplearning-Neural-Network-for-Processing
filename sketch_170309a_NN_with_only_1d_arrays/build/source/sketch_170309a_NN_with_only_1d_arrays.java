import processing.core.*; 
import processing.data.*; 
import processing.event.*; 
import processing.opengl.*; 

import java.util.HashMap; 
import java.util.ArrayList; 
import java.io.File; 
import java.io.BufferedReader; 
import java.io.PrintWriter; 
import java.io.InputStream; 
import java.io.OutputStream; 
import java.io.IOException; 

public class sketch_170309a_NN_with_only_1d_arrays extends PApplet {

Network nn;
float[] testErrorList=new float[0];


public void setup() {

  nn = new Network();
  nn.setMemoryLength(8); // doesn't do anything for lstmLight-Cells though... (except if i reset it somehow when a new pattern starts?? -> things to think about! )
  nn.addLayer(01, 50);
  nn.addLayer(50, 30);
  nn.addLayer(30, 20);
  nn.addLayer(20, 10);
  nn.addLayer(10, 02);

  nn.lr = 1e-5f;
  nn.batchSize = 20;
  //buildData();
  // buildLottoData();
   importData();

  size(1910, 500);
  strokeCap(RECT);
  strokeWeight(2);
  frameRate(100000);
}

public void draw() {

  float err = nn.learn(trainingInput, trainingTarget);
  //  nn.lr = 1*err*1e-6;
  frame.setTitle("error: "+nf(err, 1, 6)+" // "+(int)frameRate+" cps");
  if (frameCount%100!=0)return;

  float[] ret;
  background(255);
  strokeWeight(2);
  nn.resetStates();
  float absErr = 0;
  stroke(0);
  strokeWeight(1);
  line(0, height/2, width, height/2);
  strokeWeight(2);
  nn.resetStates();
  for (int i=0; i<evalInput.length; i++) {
    ret = nn.forward(evalInput[i]);
    fill(0, 200, 0);
    stroke(0);
    if (i!=0)
      //      line(3+(i-1)*6
      line(3+i*6,
      map(evalInput[i][0], min(evalInput), max(evalInput), height/6, 0),
      3+i*6,
      height/12
        );



    // stroke(100, 100, 0);
    stroke(0);
    line(3+i*6, map(evalTarget[i][0], 2, -2, 0, height), 3+i*6, map(evalTarget[i][1], 2, -2, 0, height));

    stroke(200, 0, 0);
    line(6+i*6, map(ret[0], 2, -2, 0, height), 6+i*6, map(ret[1], 2, -2, 0, height));
    // stroke(0, 200, 0);
    // line(6+i*6, height/2, 6+i*6, map(ret[1], 2, -2, 0, height));
    absErr+=abs(evalTarget[i][0]-ret[0]);
    absErr+=abs(evalTarget[i][1]-ret[1]);
  }
  testErrorList = append(testErrorList, absErr/evalInput.length);

  stroke(0);
  fill(0);
  nn.displayError();
  displayError();
}

//------------------------------------------
public void keyPressed() {
  if (key=='+') nn.lr*=10;
  if (key=='-') nn.lr*=0.1f;
  if(key=='b') {
    if(nn.batchSize!=0)nn.batchSize=0;
    else nn.batchSize = 20;
  }
  println(nn.lr);
}

//------------------------------------------
public void displayError() {
  stroke(0, 200, 0);
  strokeWeight(1);
  for (int i=0; i<testErrorList.length-1; i++)
    line(
    map(i, 0, testErrorList.length-1, 0, width),
    map(testErrorList[i], 0, max(testErrorList), height, 0),
    map(i+1, 0, testErrorList.length-1, 0, width),
    map(testErrorList[i+1], 0, max(testErrorList), height, 0)
      );
}
public float min(float[][] ar) {
  float m = 100000000;
  for (int i=0; i<ar.length; i++)for (int j=0; j<ar[i].length; j++) if (ar[i][j]<m)m=ar[i][j];
  return m;
}
public float max(float[][] ar) {
  float m = -100000000;
  for (int i=0; i<ar.length; i++)for (int j=0; j<ar[i].length; j++) if (ar[i][j]>m)m=ar[i][j];
  return m;
}
//  not used, because not needed
//---------------------------------------------------------
// the basic layer class!
abstract class BaseCell {
  // variables
 
  // functions
  abstract public float forward(float[] _x);
  abstract public float[] backward(float error) ;
  abstract public void update(float lr);
  abstract public void resetStates();
}

//  how the fuck would i do that!?^^
//---------------------------------------------------------
// the basic layer class!
abstract class BaseLayer {
  // variables
  
  // functions
  abstract public float[] forward(float[] _x);
  abstract public float[] backward(float[] error) ;
  abstract public void update(float lr);
  abstract public void resetStates();
}



//================================
// a simplified LSTM cell ("mix-gate-cell")
//
//================================
class LSTMCellLight extends  BaseCell{ //  -> maybe do that with layers instead?^^ ... idk... 
  float[] x,dx;
  float Sx, Sh;
  float G, dG;
  float dSx, dSh;
  float[] Wm, dWm, Wx, dWx;
  float B, dB;
  float Bx, dBx;
  float ShPrev;
  int numInputs;
  float learningRate = 1e-3f;

  //================================
  LSTMCellLight(int num, int t) {
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
  private float softSign(float x) {return 2.0f * x / (1.0f + abs(x));  }
  //================================
  private float dSoftSign(float x) {return 2.0f / sq(1.0f + abs(x));  }
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


//================================
class MemoryCell {
  float[][] x;
  float Sx;
  float Sh;
  float dSh;
  float dSx;
  float[] dx;
  float[][] Wx, dWx;
  float Bx, dBx;
  int numInputs;
  float weightSize   = 1.0f;
  boolean generalisation = false;
  int steps, last;
  float ret;
  boolean outputLayer=false;
  //================================
  MemoryCell(int num, int t) {
    if (num<1) num=1;
    steps     = t;
    last      = t-1;
    numInputs = num;
    weightSize = .1f; // 1.0 / numInputs;
    initArrays();
  }
  //================================
  public void initArrays() {

    x     = new float[steps][numInputs];
    dx    = new float[numInputs];
    Wx    = new float[steps][numInputs];
    dWx   = new float[steps][numInputs];
    for (int t=0; t<steps; t++) {
      for (int i=0; i<numInputs; i++) {
        Wx[t][i] =initWeight(weightSize);
      }
    }
  }
  //================================
  // 
  public float forward(float[] _x) {
    push(x, _x);  
    //  orig:
    Sx = 0;
    for (int t = 0; t<steps; t++) {
      for (int in=0; in<numInputs; in++) {
        Sx += x[t][in] * Wx[t][in];
      }
    }
    Sx += Bx;
    Sh = softSign(Sx);
    return Sh;
  }
  //================================
  public float[] backward(float error) {
    dx = new float[dx.length];  

    //dSx = Sx * (1 - Sx ) * error;       // sigmoid
    dSx = (2.0f / sq(1.0f + abs(Sx))) * error;         // Softsign
    //dSx = (1 - sq(Sx)) * error;       // tanh
    dBx += dSx;
    for (int t=steps-1; t>=0; t--) {
      for (int in=0; in<numInputs; in++) {
        dx[in]     += Wx[t][in]*dSx;
        dWx[t][in] += x[t][in] *dSx;
      }
    }
    return dx;
  }
  //================================
  public void update(float learningRate) {
    for (int t=0; t<steps; t++) {
      for (int i=0; i<Wx[t].length; i++) {
        Wx[t][i]  += dWx[t][i]  * learningRate;
      }
    }
    Bx += dBx * learningRate;
    // generalize:
    for (int t=0; t<steps; t++) {
      for (int i=0; i<Wx[t].length; i++) {
        Wx[t][i]  *=(1-learningRate*0.01f);
      }
    }

    resetGradients();
    resetStates(); // why?
  }
  //================================
  private void resetGradients() {
    dWx = new float[dWx.length][dWx[0].length];
    dBx = 0;
  }
  //================================
  public void resetStates() {
    x = new float[steps][numInputs];
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
  private float sigmoid(float val) {
    return 1 / (1 + exp(-1 * val));
  }
  //================================
  private float tanh(float x) { 
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
  }
  //================================
  private float rectifiedLinear(float x) {
    return max(x, 0.0f);
  }
  //================================
  private float softSign(float x) {
    return 2 * x / (1 + abs(x));
  }

  //================================
  private float coinflip() {
    if (random(100)>=50) return 1;
    else return -1;
  }
  //================================
  private float initWeight(float w) {
    return random(-w, w);
  }
  //================================
  public float averageWeightSize() {
    return absSum(Wx) / (steps*numInputs);
  }
  //================================
  private float absSum(float[] ar) {
    ret=0;
    for (int i=0; i<ar.length; i++)
      ret+=abs(ar[i]);
    return ret;
  }
  //================================
  private float absSum(float[][] ar) {
    ret = 0;
    for (int i=0; i<ar.length; i++)
      for (int j=0; j<ar[i].length; j++)
        ret += abs(ar[i][j]);
    return ret;
  }
  //================================
}

class SimpleCell {
  float[] Wx;
  float Bx;
  float Sy;
  float[] Sx;
  float[] dWx;
  float dBx;
  float[] dSx;
  //------------------------------
  SimpleCell(int in, int tmp) {
    Wx= new float[in];
    dWx = new float[in];
    Sx = new float[in];
    dSx = new float[in];
    for (int i=0; i<Wx.length; i++)
      Wx[i] = random(.051f, .2f)*random(100)>50?-1:1;
  }
  //------------------------------
  public float forward(float[] in) {

    Sx = in;
    Sy = 0;
    for (int i=0; i<in.length; i++)
      Sy+=(Wx[i])*in[i];
    Sy+=Bx;
    Sy= softSign(Sy);
    return Sy;
  }
  public void resetStates(){}

  //------------------------------
  public float[] backward(float dSy) {
    dSy = (2.0f / sq(1.0f+abs(Sy)))*dSy; // softsign
    dBx+=dSy;
    dSx = new float[dSx.length];
    for (int i=0; i<Wx.length; i++) {
      dWx[i] += Sx[i]*dSy;
      dSx[i] += Wx[i]*dSy;
    }
    return dSx;
  }
  //------------------------------
  public void update(float lr) {
    for (int i=0; i<Wx.length; i++) {
      Wx[i]+=dWx[i]*lr;
    }
    Bx+=dBx*lr;
    dWx=new float[dWx.length];
    dBx=0;
  }
  //------------------------------
  public float softSign(float x) {
    return 2.0f * x / (1.0f + abs(x));
  }
  //------------------------------
}

float inputData[][], targetData[][];


float trainingInput[][], trainingTarget[][], evalInput[][], evalTarget[][];


//------------------------------------------
public void importData() {
  int numTrainingSamples = 1000;
  int numEvalSamples = 300;
  int futureOffset = 3;


  String lines[] = loadStrings("EURUSD1440.csv");
  String tmp[][] = new String[lines.length][0];
  //  println(lines.length+" entries in input file");
  // prepare data array;
  for (int i=0; i<lines.length; i++)
    tmp[i] = split(lines[i], ",");

  tmp = (String[][]) subset(tmp, 1000);
  float[][] data;
  float[][] f, t;

  data = new float[tmp.length][tmp[0].length];
  for (int i=0; i<tmp.length; i++)
    for (int j=0; j<tmp[i].length-2; j++) {
      data[i][j] = Float.parseFloat(tmp[i][j+2]);
    }
  f = new float[numTrainingSamples][1];
  t = new float[f.length][2];

  // build training data

  for (int i=1; i<f.length-futureOffset; i++) {
    //f[i][0] =  percentChange(data[i-1][1], data[i][1]); // high
    // f[i][1] =  percentChange(data[i-1][2], data[i][2]); // low
    // f[i][2] =  percentChange(data[i-1][3], data[i][3]); // close
    f[i][0] =  percentChange(data[i-1][3], data[i][3]); // close

    //  t[i][0] = percentChange(data[i][1], maximum(data, 1, futureOffset, i));
    //  t[i][1] = percentChange(data[i][2], minimum(data, 2, futureOffset, i));


    t[i][0] = percentChange(data[i][3], minimum(data, 3, futureOffset, i))* 0.2f;
    t[i][1] = percentChange(data[i][3], maximum(data, 3, futureOffset, i))* 0.2f;
  }
  trainingInput = f;
  trainingTarget = t;


  // build evaluationData
  tmp = (String[][]) subset(tmp, 1*numTrainingSamples);

  data = new float[tmp.length][tmp[0].length];
  for (int i=0; i<tmp.length; i++)
    for (int j=0; j<tmp[i].length-2; j++) {
      data[i][j] = Float.parseFloat(tmp[i][j+2]);
    }
  f = new float[numEvalSamples][1];
  t = new float[f.length][2];

  // build training data

  for (int i=1; i<f.length-futureOffset; i++) {
    //  f[i][0] =  percentChange(data[i-1][1], data[i][1]); // high
    // f[i][1] =  percentChange(data[i-1][2], data[i][2]); // low
    // f[i][2] =  percentChange(data[i-1][3], data[i][3]); // close
    f[i][0] =  percentChange(data[i-1][3], data[i][3]); // close



    // t[i][0] = percentChange(data[i][1], maximum(data, 1, futureOffset, i));
    // t[i][1] = percentChange(data[i][2], minimum(data, 2, futureOffset, i));

    t[i][0] = percentChange(data[i][3], minimum(data, 3, futureOffset, i))* 0.2f;
    t[i][1] = percentChange(data[i][3], maximum(data, 3, futureOffset, i))* 0.2f;
  }
  evalInput = f;
  evalTarget = t;
  if (true) {
    evalInput = trainingInput;
    evalTarget = trainingTarget;
  }
}
//------------------------------------------
public float minimum(float[][] ar, int index, int period, int pos) {
  float thisMin = 10000000;
  for (int i=1; i<period+1; i++) {
    if (ar[pos+i][index]<thisMin) thisMin=ar[pos+i][index];
  }
  return thisMin;
}
//------------------------------------------
public float maximum(float[][] ar, int index, int period, int pos) {
  float thisMax = -10000000;
  for (int i=1; i<period+1; i++) {
    if (ar[pos+i][index]>thisMax) thisMax=ar[pos+i][index];
  }
  return thisMax;
}
//------------------------------------------
public float percentChange(float a, float b) {
  return ( a==0 ? 0 : 100.0f*(b-a)/a );
}
//------------------------------------------
public void buildData() {
  int numSamples = 350;
  inputData=new float[numSamples][1];
  targetData = new float[numSamples][2];
  for (int i=1; i<inputData.length; i++) {
    for (int j=0; j<inputData[i].length; j++) {
      inputData[i][j] = sin(i*.1f); // inputData[i-1][j]+random(-0.2, 0.2);
    }
  }
  for (int i=0; i<inputData.length-1; i++) {
    targetData[i][0] =cos(i*.2f);
    targetData[i][1] =-sin(i*.1f);
  }


  trainingInput = inputData;
  trainingTarget = targetData;

  if (true) {
    inputData=new float[numSamples][1];
    targetData = new float[numSamples][2];
    int index=0;

    for (int i=1; i<inputData.length; i++) {
      index = i+10;
      for (int j=0; j<inputData[i].length; j++) {
        inputData[i][j] = sin(index*.1f); // inputData[i-1][j]+random(-0.2, 0.2);
      }
    }
    for (int i=0; i<inputData.length-1; i++) {
      index = i+10;
      targetData[i][0] =1*cos(index*.2f);
      targetData[i][1] =-1*sin(index*.2f);
    }
  }

  evalInput = inputData;
  evalTarget = targetData;

  evalInput = trainingInput;
  evalTarget = trainingTarget;

  /*
  // XOR
   inputData[0][0] = 0;
   inputData[0][1] = 0;
   inputData[1][0] = 0;
   inputData[1][1] = 1;
   inputData[2][0] = 1;
   inputData[2][1] = 0;
   inputData[3][0] = 1;
   inputData[3][1] = 1;
   targetData[0][0] = 1;
   targetData[1][0] = 0;
   targetData[2][0] = 0;
   targetData[3][0] = 1;
   */
}











public void buildLottoData() {
  int numSamples = 300;
  inputData=new float[numSamples][6];
  targetData = new float[numSamples][6];
  for (int i=0; i<inputData.length; i++) {
    for (int j=0; j<inputData[i].length; j++) {
      inputData[i][j] = ((int)random(45))/45.0f;
    }
  }
  for (int i=0; i<inputData.length-2; i++) {
    for (int j=0; j<inputData[i].length; j++) {
      targetData[i][j] =inputData[i+1][j];
    }
  }

  trainingInput = inputData;
  trainingTarget = targetData;

  if (true) {
    inputData=new float[numSamples][1];
    targetData = new float[numSamples][2];
    int index=0;

    for (int i=0; i<inputData.length; i++) {
      for (int j=0; j<inputData[i].length; j++) {
        inputData[i][j] = ((int)random(45))/45.0f;
      }
    }
    for (int i=0; i<inputData.length-2; i++) {
      for (int j=0; j<inputData[i].length; j++) {
        targetData[i][j] =inputData[i+1][j];
      }

      evalInput = inputData;
      evalTarget = targetData;
    }
  }
}
//------------------------------
class Layer {
  //BaseCell[] cells;
  MemoryCell[] cells;
  float[] Sy;
  float[] dSx;
  int memory = 10;
  //------------------------------
  Layer(int in, int out, int memoryLength) {
    Sy = new float[out];
    dSx = new float[in];
    cells = new MemoryCell[out];
    memory = memoryLength;
    for (int i=0; i<out; i++) {
      cells[i] = new MemoryCell(in, memory);
    }
  }
  //------------------------------
  public void forward(float[] in) {
    for (int i=0; i<cells.length; i++) {
      Sy[i] = cells[i].forward(in);
    }
  }
  //------------------------------
  public void backward(float[] dy) {
    dSx = new float[dSx.length];
    float[] tmp = new float[dSx.length];
    for (int i=0; i<cells.length; i++) {
      tmp = cells[i].backward(dy[i]);
      for (int j=0; j<tmp.length; j++) {
        dSx[j]+=tmp[j];
      }
    }
  }
  //------------------------------
  public void update(float lr) {
    for (int i=0; i<cells.length; i++)
      cells[i].update(lr);
  }
  //------------------------------
  public void setMemoryLength(int l) {
    memory = l;
  }
  public void resetStates() {
    for (int i=0; i<cells.length; i++)
      cells[i].resetStates();
  }
}

//------------------------------
class Network {
  Layer[] layers;
  float lr = 1e-4f;
  float[] errorList=new float[0];
  float[] memory;
  int memoryLength = 10;
  int batchSize = 0;
  //------------------------------
  Network() {
    layers = new Layer[0];
  memory = new float[50];
  }
  //------------------------------
  public void setMemoryLength(int l) {    
    memory = new float[l];
   memoryLength = l;
  }
  //------------------------------
  public void resetMemory() {
    memory = new float[memory.length];
  }
  public void resetStates(){
    for(int i=0;i<layers.length;i++)
    layers[i].resetStates();
  }
  //------------------------------
  public float learn(float[][] in, float[][] out) {
    float[] ret = new float[1];
    float[] err = new float[out[0].length];
    resetStates();
//    memory = new float[memory.length];
    float absErr = 0;

    for (int i=0; i<in.length; i++) {
      ret = forward(in[i]);

      for (int j=0; j<ret.length; j++) {
        err[j] = out[i][j]-ret[j];
        absErr+=abs(err[j]);
        err[j]*=abs(err[j]); // error squared!
      }
      backward(err);
      if(batchSize!=0 && i%batchSize==0) update(); // batchlearn
    }
    update();
    if (frameCount%100==0) {
      errorList = append(errorList, absErr/in.length);
    }
    return absErr/in.length;
  }

  //------------------------------
  public void push(float[] ar, float val) {
    for (int i=0; i<ar.length-1; i++) {
      ar[i+1]=ar[i];
    }
    ar[0]=val;
  }
  
  //------------------------------
  public float[] forward(float[] in) {
  //  push(memory, in[0]);
    layers[0].forward(in);
    for (int i=1; i<layers.length; i++) {
      layers[i].forward(layers[i-1].Sy);
    }
    return layers[layers.length-1].Sy;
  }
  //------------------------------
  public void backward(float[] err) {
    layers[layers.length-1].backward(err);
    for (int i=layers.length-2; i>=0; i--) {
      layers[i].backward(layers[i+1].dSx);
    }
  }
  //------------------------------
  public void update() {
    for (int i=0; i<layers.length; i++)
      layers[i].update(lr);
  }
  //------------------------------
  public void addLayer(int in, int out) {
    layers =  (Layer[]) expand(layers, layers.length+1);
    layers[layers.length-1] = new Layer(in, out,memoryLength);
  }
  //------------------------------
  public void displayError() {
    stroke(200, 0, 0);
    strokeWeight(1);
    for (int i=0; i<this.errorList.length-1; i++)
      line(
      map(i, 0, errorList.length-1, 0, width), 
      map(errorList[i], 0, max(errorList), height, 0), 
      map(i+1, 0, errorList.length-1, 0, width), 
      map(errorList[i+1], 0, max(errorList), height, 0)
        );
  }
  //------------------------------
}

  static public void main(String[] passedArgs) {
    String[] appletArgs = new String[] { "sketch_170309a_NN_with_only_1d_arrays" };
    if (passedArgs != null) {
      PApplet.main(concat(appletArgs, passedArgs));
    } else {
      PApplet.main(appletArgs);
    }
  }
}
