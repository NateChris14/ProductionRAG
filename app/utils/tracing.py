"""
OpenTelemetry + LangFuse tracing setup.
Call configure_tracing() once at app startup in main.py lifespan.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.oltp.proto.grpc.trace_exporter import OLTPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from langfuse import Langfuse
from app.config import get_settings

_settings = get_settings()

def configure_tracing(app=None) -> None:
    
    provider = TracerProvider()
    exporter = OLTPSpanExporter(
        endpoint=_settings.otel_exporter_otlp_endpoint
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    if app:
        FastAPIInstrumentor.instrument_app(app)

    # Langfuse is opt-in - only intialised if keys are present
    if _settings.langfuse_public_key:

        Langfuse(
            public_key=_settings.langfuse_public_key,
            secret_key=_settings.langfuse_secret_key,
            host=_settings.langfuse_host
        )

def get_tracer(name: str = "rag") -> trace.Tracer:
    return trace.get_tracer(name)


