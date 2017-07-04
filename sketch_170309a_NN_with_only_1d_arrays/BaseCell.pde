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

