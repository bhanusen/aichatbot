import os
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

def setup_evaluation():
    """
    Initializes OpenTelemetry to send traces to a local Arize Phoenix server.
    Returns the URL of the Phoenix dashboard.
    """
    # Configure OpenTelemetry to send traces to Phoenix
    endpoint = "http://127.0.0.1:6007/v1/traces"
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    
    # Instrument LangChain
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    
    return "http://127.0.0.1:6007"
