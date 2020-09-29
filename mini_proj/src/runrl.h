#ifndef __RUNRL_H__
#define __RUNRL_H__

class RLRunnable
{
public:
  virtual void runRL() = 0;
  virtual double scoreRL() = 0;
};

#endif
