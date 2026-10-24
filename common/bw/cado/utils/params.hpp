/* bw-local stub (see common/bw/shims/; original at cado b5ca9c4
 * utils/params.hpp). The driver fills lingen_hints directly, so the
 * parameter list is opaque and all its methods inert. */
#ifndef _CADO_UTILS_PARAMS_HPP_
#define _CADO_UTILS_PARAMS_HPP_
#include <string>
namespace cado {
namespace params {
class cxx_param_list {
public:
    void declare_usage(const char *, const char *) {}
    void lookup(const char *) {}
    template<class T> bool parse(const char *, T &) { return false; }
    template<class T> bool parse(const char *, T &, const char *) { return false; }
};
/* the cxx_mpz parser specializations remain uncompiled */
template<class T> struct parser { typedef T type; };
} /* namespace params */
}
using cxx_param_list = cado::params::cxx_param_list;
#endif
