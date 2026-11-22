#pragma once
#include "mfem.hpp"

using namespace mfem;

class LogarithmGridFunctionCoefficient : public Coefficient
{
protected:
  GridFunction *u; // grid function
  Coefficient *obstacle;
  real_t min_val;

public:
  LogarithmGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_,
                                   real_t min_val_ = -36)
      : u(&u_), obstacle(&obst_), min_val(min_val_) {}

  real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    MFEM_ASSERT(u != NULL, "grid function is not set");

    real_t val = u->GetValue(T, ip) - obstacle->Eval(T, ip);
    return std::max(min_val, log(val));
  }
};

class ExponentialGridFunctionCoefficient : public Coefficient
{
protected:
  GridFunction *u;
  Coefficient *obstacle;
  real_t min_val;
  real_t max_val;

public:
  ExponentialGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_,
                                     real_t min_val_ = 0.0,
                                     real_t max_val_ = 1e6)
      : u(&u_), obstacle(&obst_), min_val(min_val_), max_val(max_val_) {}

  real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    MFEM_ASSERT(u != NULL, "grid function is not set");

    real_t val = u->GetValue(T, ip);
    return std::min(max_val, std::max(min_val, exp(val) + obstacle->Eval(T, ip)));
  }
};