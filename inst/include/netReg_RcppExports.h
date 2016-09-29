// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#ifndef RCPP_netReg_RCPPEXPORTS_H_GEN_
#define RCPP_netReg_RCPPEXPORTS_H_GEN_

#include <RcppArmadillo.h>
#include <Rcpp.h>

namespace netReg {

    using namespace Rcpp;

    namespace {
        void validateSignature(const char* sig) {
            Rcpp::Function require = Rcpp::Environment::base_env()["require"];
            require("netReg", Rcpp::Named("quietly") = true);
            typedef int(*Ptr_validate)(const char*);
            static Ptr_validate p_validate = (Ptr_validate)
                R_GetCCallable("netReg", "netReg_RcppExport_validate");
            if (!p_validate(sig)) {
                throw Rcpp::function_not_exported(
                    "C++ function with signature '" + std::string(sig) + "' not found in netReg");
            }
        }
    }

    inline Rcpp::List _edgenet_cpp(const Rcpp::NumericMatrix& XS, const Rcpp::NumericMatrix& YS, const Rcpp::NumericMatrix& GXS, const Rcpp::NumericMatrix& GYS, const int n, const int p, const int q, const double lambda, const double psigx, const double psigy, const int n_iter, const double thresh, const Rcpp::CharacterVector& family) {
        typedef SEXP(*Ptr__edgenet_cpp)(SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP);
        static Ptr__edgenet_cpp p__edgenet_cpp = NULL;
        if (p__edgenet_cpp == NULL) {
            validateSignature("Rcpp::List(*_edgenet_cpp)(const Rcpp::NumericMatrix&,const Rcpp::NumericMatrix&,const Rcpp::NumericMatrix&,const Rcpp::NumericMatrix&,const int,const int,const int,const double,const double,const double,const int,const double,const Rcpp::CharacterVector&)");
            p__edgenet_cpp = (Ptr__edgenet_cpp)R_GetCCallable("netReg", "netReg__edgenet_cpp");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p__edgenet_cpp(Rcpp::wrap(XS), Rcpp::wrap(YS), Rcpp::wrap(GXS), Rcpp::wrap(GYS), Rcpp::wrap(n), Rcpp::wrap(p), Rcpp::wrap(q), Rcpp::wrap(lambda), Rcpp::wrap(psigx), Rcpp::wrap(psigy), Rcpp::wrap(n_iter), Rcpp::wrap(thresh), Rcpp::wrap(family));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<Rcpp::List >(rcpp_result_gen);
    }

}

#endif // RCPP_netReg_RCPPEXPORTS_H_GEN_
