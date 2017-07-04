
float inputData[][], targetData[][];


float trainingInput[][], trainingTarget[][], evalInput[][], evalTarget[][];


//------------------------------------------
void importData() {
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


    t[i][0] = percentChange(data[i][3], minimum(data, 3, futureOffset, i))* 0.2;
    t[i][1] = percentChange(data[i][3], maximum(data, 3, futureOffset, i))* 0.2;
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

    t[i][0] = percentChange(data[i][3], minimum(data, 3, futureOffset, i))* 0.2;
    t[i][1] = percentChange(data[i][3], maximum(data, 3, futureOffset, i))* 0.2;
  }
  evalInput = f;
  evalTarget = t;
  if (true) {
    evalInput = trainingInput;
    evalTarget = trainingTarget;
  }
}
//------------------------------------------
float minimum(float[][] ar, int index, int period, int pos) {
  float thisMin = 10000000;
  for (int i=1; i<period+1; i++) {
    if (ar[pos+i][index]<thisMin) thisMin=ar[pos+i][index];
  }
  return thisMin;
}
//------------------------------------------
float maximum(float[][] ar, int index, int period, int pos) {
  float thisMax = -10000000;
  for (int i=1; i<period+1; i++) {
    if (ar[pos+i][index]>thisMax) thisMax=ar[pos+i][index];
  }
  return thisMax;
}
//------------------------------------------
float percentChange(float a, float b) {
  return ( a==0 ? 0 : 100.0*(b-a)/a );
}
//------------------------------------------
void buildData() {
  int numSamples = 350;
  inputData=new float[numSamples][1];
  targetData = new float[numSamples][2];
  for (int i=1; i<inputData.length; i++) {
    for (int j=0; j<inputData[i].length; j++) {
      inputData[i][j] = sin(i*.1); // inputData[i-1][j]+random(-0.2, 0.2);
    }
  }
  for (int i=0; i<inputData.length-1; i++) {
    targetData[i][0] =cos(i*.2);
    targetData[i][1] =-sin(i*.1);
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
        inputData[i][j] = sin(index*.1); // inputData[i-1][j]+random(-0.2, 0.2);
      }
    }
    for (int i=0; i<inputData.length-1; i++) {
      index = i+10;
      targetData[i][0] =1*cos(index*.2);
      targetData[i][1] =-1*sin(index*.2);
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











void buildLottoData() {
  int numSamples = 300;
  inputData=new float[numSamples][6];
  targetData = new float[numSamples][6];
  for (int i=0; i<inputData.length; i++) {
    for (int j=0; j<inputData[i].length; j++) {
      inputData[i][j] = ((int)random(45))/45.0;
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
        inputData[i][j] = ((int)random(45))/45.0;
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
