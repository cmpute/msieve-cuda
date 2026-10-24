/* bw-local shim: the vendored lingen keeps its parameter plumbing in
 * decl_usage/lookup_parameters signatures only; this driver fills the
 * lingen_hints directly, so the list itself is an opaque stub. */
#ifndef _PARAMS_HPP_
#define _PARAMS_HPP_
#include <string>
#include <vector>
class cxx_param_list {
public:
    /* the signatures the vendored code calls; the driver never
       parses command lines, so these are all inert */
    void declare_usage(const char *, const char *) {}
    void lookup(const char *) {}
    template<class T> void parse(const char *, T &) {}
};
#endif
