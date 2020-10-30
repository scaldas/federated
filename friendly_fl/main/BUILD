load("@rules_python//python:defs.bzl", "py_binary", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "federated_trainer",
    srcs = ["federated_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//compression:sparsity",
        "//friendly_fl/emnist:federated_emnist",
        "//friendly_fl/shakespeare:federated_shakespeare",
        "//utils:utils_impl",
    ],
)
