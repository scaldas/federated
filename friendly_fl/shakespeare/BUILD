load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "federated_shakespeare",
    srcs = ["federated_shakespeare.py"],
    srcs_version = "PY3",
    deps = [
        "//friendly_fl/shared:keras_metrics",
        "//utils:training_loop",
        "//utils:training_utils",
        "//utils/datasets:shakespeare_dataset",
        "//utils/models:shakespeare_models",
    ],
)


