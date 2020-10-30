load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "keras_metrics",
    srcs = ["keras_metrics.py"],
    srcs_version = "PY3",
)

py_test(
    name = "keras_metrics_test",
    size = "small",
    srcs = ["keras_metrics_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":keras_metrics"],
)