"""FastAPI router packages for ZI_RAG sidecar."""

from .admin import router as admin_router
from .analyze import router as analyze_router
from .chat_attachments import router as chat_attachments_router
from .compliance import router as compliance_router
from .documents import router as documents_router
from .indexes import router as indexes_router
from .jobs import router as jobs_router

__all__ = [
    "admin_router",
    "analyze_router",
    "chat_attachments_router",
    "compliance_router",
    "documents_router",
    "indexes_router",
    "jobs_router",
]
