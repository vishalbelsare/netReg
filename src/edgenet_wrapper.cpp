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

#include "edgenet_wrapper.hpp"
#include "edgenet_gaussian.hpp"
#include "stat_functions.hpp"

#include <numeric>
#include <vector>
#include <string>
#include <map>

namespace netreg
{
    SEXP edgenet_wrapper::run(graph_penalized_linear_model_data &data) const
    {
        BEGIN_RCPP
        netreg::edgenet_gaussian edge;
        arma::Mat<double> coef = edge.run(data);
        arma::Col<double> intr = intercept(data.design(),
                                           data.response(),
                                           coef);

        return Rcpp::List::create(
            Rcpp::Named("coefficients") = coef,
            Rcpp::Named("intercept") = intr
        );
        END_RCPP
        return R_NilValue;
    }

    SEXP regularization_path
        (graph_penalized_linear_model_cv_data &data)
    {
        BEGIN_RCPP
        netreg::edgenet_gaussian_model_selection edge;
        std::map<std::string, double> res = edge.regularization_path(data);

        return Rcpp::List::create(
            Rcpp::Named("parameters") = Rcpp::wrap(res),
            Rcpp::Named("folds") = data.fold_ids()
        );
        END_RCPP
        return R_NilValue;
    }
}
