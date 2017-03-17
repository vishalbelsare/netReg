/**
 * netReg: graph-regularized linear regression models.
 * <p>
 * Copyright (C) 2015 - 2016 Simon Dirmeier
 * <p>
 * This file is part of netReg.
 * <p>
 * netReg is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * netReg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with netReg. If not, see <http://www.gnu.org/licenses/>.
 *
 * @author: Simon Dirmeier
 * @email: simon.dirmeier@gmx.de
 */

#ifndef NETREG_GRAPH_FUNCTIONS_HPP
#define NETREG_GRAPH_FUNCTIONS_HPP

#ifdef USE_RCPPARMADILLO
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#else
#include <Eigen/Dense>
#endif

namespace netreg
{
   /*
    * Calculate the normalized laplacian of a matrix.
    *
    * @param x the pointer for which the laplacian is calculated (col first)
    * @param n nrows of x
    * @param m ncols of y
    * @return the normalized laplacian
    */
   Eigen::MatrixXd laplacian(const double * x, int n, int m, double px);

}
#endif //NETREG_GRAPH_FUNCTIONS_HPP
