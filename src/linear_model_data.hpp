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

#ifndef NETREG_LINEAR_MODEL_DATA_HPP
#define NETREG_LINEAR_MODEL_DATA_HPP

#include <vector>
#include <string>
#include <utility>

#ifdef USE_RCPPARMADILLO
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#else
#include <Eigen/Dense>
#endif
#include "family.hpp"

namespace netreg
{
    /**
     * Data-structure for all required data for a linear model
     */
    class linear_model_data
    {
    protected:

        /**
         * Protected constructor in order to avoid instantiation.
         *
         * @param x the design matrix as pointer
         * @param y the desponse matrix as pointer
         * @param n number of samples
         * @param p number of covariables
         * @param q number of responses
         * @param niter maximal number of iterations
         * @param thresh the convergence threshold
         * @param fam of distribution of y
         */
        linear_model_data(double *x, double *y,
                          const int n, const int p, const int q,
                          const int niter, const double thresh,
                          const family fam)
            : N(n), P(p), Q(q),
              X(x, n, p),
              Y(y, n, q),
              THRESH(thresh), N_ITER(niter),
              TXY(p, q), txx_rows_(p),
              family_(fam)
        {
            Eigen::MatrixXd TX = X.transpose();
            Eigen::MatrixXd TXX = TX * X;
            TXY = TX * Y;
            for(std::vector< Eigen::RowVectorXd >::size_type i = 0; i < TXX.rows();  ++i)
            {
                txx_rows_[i] = TXX.row(i);
            }
        }

    public:

        /**
         * Getter for the family of the distribution of the response.
         *
         * @return returns the family
         */
        const family distribution_family()
        {
            return family_;
        }

        /**
         * Getter for number of samples.
         *
         * @return returns the number of samples
         */
        const int sample_count()
        {
            return N;
        }

        /**
         * Getter for number of responses.
         *
         * @return returns the number of responses
         */
        const int response_count()
        {
            return Q;
        }

        /**
         * Getter for number of covariables.
         *
         * @return returns the number of covariables
         */
        const int covariable_count()
        {
            return P;
        }

        Eigen::RowVectorXd& txx_row(const int i)
        {
            return txx_rows_[i];
        }

        std::vector< Eigen::RowVectorXd >& txx_rows()
        {
            return txx_rows_;
        }

        /**
         * Getter for the design matrix.
         *
         * @return a reference to the design matrix
         */
        Eigen::MatrixXd& design()
        {
            return X;
        }

        /**
         * Getter for the response matrix.
         *
         * @return a reference to the response matrix
         */
        Eigen::MatrixXd& response()
        {
            return Y;
        }

        /**
         * Getter for X'Y matrix.
         *
         * @return a reference to X'Y matrix.
         */
        Eigen::MatrixXd& txy()
        {
            return TXY;
        }

        /**
         * Getter for the number of maximal iterations.
         *
         * @return the max number of iterations
         */
        const int max_iter()
        {
            return N_ITER;
        }

        /**
         * Getter for the convergence threshold of CCD.
         *
         * @return the convergence threshold.
         */
        double threshold()
        {
            return THRESH;
        }

    protected:
        const int N;             // number of samples: n
        const int P;             // number of covariables: p
        const int Q;             // number of responses: q
        Eigen::MatrixXd X;        // (n x p)-dimensional design matrix
        Eigen::MatrixXd Y;        // (n x q)-dimensional response matrix
        const double THRESH;     // convergence threshold
        const int N_ITER;        // max number iterations if CCD does not converge
        Eigen::MatrixXd TXY;      // (p x q)-dimensional matrix: X'Y
        std::vector< Eigen::RowVectorXd& > txx_rows_;
        const enum family family_;      // family of distribution of y
    };
}
#endif //NETREG_LINEARMODELDATA_HPP
