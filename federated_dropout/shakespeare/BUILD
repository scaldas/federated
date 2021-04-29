load("@rules_python//python:defs.bzl", "py_binary", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_binary(
    name = "federated_shakespeare",
    srcs = ["federated_shakespeare.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":shakespeare_utils",
        "//federated_dropout/dropout_utils:simple_fedavg_tf",
        "//federated_dropout/dropout_utils:simple_fedavg_tff",
        "//federated_dropout/utils:keras_metrics"
    ],
)

py_library(
    name = "shakespeare_utils",
    srcs = ["shakespeare_utils_2_lstm.py"],
    srcs_version = "PY3",
    deps = [
        "//utils/datasets:shakespeare_dataset",
    ],
)