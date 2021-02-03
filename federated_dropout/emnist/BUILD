load("@rules_python//python:defs.bzl", "py_binary", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_binary(
    name = "federated_emnist",
    srcs = ["federated_emnist.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":emnist_utils",
        "//federated_dropout/dropout_utils:simple_fedavg_tf",
        "//federated_dropout/dropout_utils:simple_fedavg_tff",
    ],
)

py_library(
    name = "emnist_utils",
    srcs = ["emnist_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//utils/datasets:emnist_dataset",
    ],
)