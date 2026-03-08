
try:
    from prometheus_client import Counter, Histogram

    prediction_counter = Counter(
        "hsi_predictions_total",
        "Total hyperspectral predictions",
        ["task", "result"]
    )

    prediction_latency = Histogram(
        "hsi_prediction_seconds",
        "Prediction latency"
    )

    def record_prediction(task, result):
        prediction_counter.labels(task=task, result=result).inc()

    def record_latency(seconds):
        prediction_latency.observe(seconds)

except ImportError:
    import logging as _logging
    _log = _logging.getLogger(__name__)
    _log.warning("prometheus_client not installed — monitoring metrics disabled. "
                 "Install with: pip install prometheus-client")

    def record_prediction(task, result):  # type: ignore[misc]
        pass

    def record_latency(seconds):  # type: ignore[misc]
        pass
