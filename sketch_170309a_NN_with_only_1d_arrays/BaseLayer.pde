//  how the fuck would i do that!?^^ -> needs some more thinking!
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
