#include "goma.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include <math.h>

#include "mm_fill_poisson.h"
#include "rf_fem_const.h"


int assemble_poisson() {

  int eqn = R_POISSON;

  // We shouldn't call this function
  // if poisson not enabled
  if (!pd->e[eqn]) {
    return -1;
  }

  dbl g = - 10 * exp(-(SQUARE((fv->x[0] - 0.5)) + SQUARE((fv->x[1]-0.5))) / 0.02);

  if (af->Assemble_Residual) {
    int peqn = upd->ep[eqn];
    for (int i = 0; i < ei->dof[eqn]; i++) {
      dbl diffusion = 0;
      if (pd->etm[eqn][LOG2_DIFFUSION]) {
        for (int a = 0; a < pd->Num_Dim; a++) {
          diffusion += bf[eqn]->grad_phi[i][a] * fv->grad_u[a];
        }
      }
      diffusion *= -fv->wt * bf[eqn]->detJ * fv->h3;
      diffusion *= pd->etm[eqn][LOG2_DIFFUSION];

      dbl source = 0;
      if (pd->etm[eqn][LOG2_SOURCE]) {
        source += g * bf[eqn]->phi[i];
      }
      source *= -fv->wt * bf[eqn]->detJ * fv->h3;
      source *= pd->etm[eqn][LOG2_SOURCE];

      lec->R[LEC_R_INDEX(peqn, i)] += diffusion + source;
    }
  }
  
  if (af->Assemble_Jacobian) {
    int peqn = upd->ep[eqn];
    for (int i = 0; i < ei->dof[eqn]; i++) {
      int var = POISSON;
      for (int j = 0; j < ei->dof[eqn]; j++) {
        int pvar = upd->vp[var];
        dbl diffusion = 0;
        if (pd->etm[eqn][LOG2_DIFFUSION]) {
          for (int a = 0; a < pd->Num_Dim; a++) {
            diffusion += bf[eqn]->grad_phi[i][a] * bf[var]->grad_phi[j][a];
          }
        }
        diffusion *= -fv->wt * bf[eqn]->detJ * fv->h3;
        diffusion *= pd->etm[eqn][LOG2_DIFFUSION];

        dbl source = 0;

        lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += diffusion + source;
      }
    }
  }

  return 0;
}

void poisson_side_sin_bc(dbl func[DIM],
                         dbl d_func[DIM][MAX_VARIABLE_TYPES + MAX_CONC][MDE],
                         dbl alpha,
                         dbl beta,
                         dbl gamma,
                         dbl omega,
                         dbl zeta)
{
  func[0] = alpha * sin(beta * fv->x[0] + gamma * fv->x[1] + omega * fv->x[2]) + zeta;

  // if we had derivatives
  // d_func[0][POISSON][j] = something * phi_j
}
