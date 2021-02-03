load("@rules_python//python:defs.bzl", "py_binary", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "simple_fedavg_tf",
    srcs = ["simple_fedavg_tf.py"],
    srcs_version = "PY3",
)

py_library(
    name = "simple_fedavg_tff",
    srcs = ["simple_fedavg_tff.py"],
    srcs_version = "PY3",
    deps = [
        ":simple_fedavg_tf",
    ],
)
