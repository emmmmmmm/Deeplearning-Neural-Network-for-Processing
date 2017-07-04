Network nn;
float[] testErrorList=new float[0];


void setup() {

  nn = new Network();
  nn.setMemoryLength(8); // doesn't do anything for lstmLight-Cells though... (except if i reset it somehow when a new pattern starts?? -> things to think about! )
  nn.addLayer(01, 50);
  nn.addLayer(50, 30);
  nn.addLayer(30, 20);
  nn.addLayer(20, 10);
  nn.addLayer(10, 02);

  nn.lr = 1e-5;
  nn.batchSize = 20;
  //buildData();
  // buildLottoData();
   importData();

  size(1910, 500);
  strokeCap(RECT);
  strokeWeight(2);
  frameRate(100000);
}

void draw() {

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
void keyPressed() {
  if (key=='+') nn.lr*=10;
  if (key=='-') nn.lr*=0.1;
  if(key=='b') {
    if(nn.batchSize!=0)nn.batchSize=0;
    else nn.batchSize = 20;
  }
  println(nn.lr);
}

//------------------------------------------
void displayError() {
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
float min(float[][] ar) {
  float m = 100000000;
  for (int i=0; i<ar.length; i++)for (int j=0; j<ar[i].length; j++) if (ar[i][j]<m)m=ar[i][j];
  return m;
}
float max(float[][] ar) {
  float m = -100000000;
  for (int i=0; i<ar.length; i++)for (int j=0; j<ar[i].length; j++) if (ar[i][j]>m)m=ar[i][j];
  return m;
}
