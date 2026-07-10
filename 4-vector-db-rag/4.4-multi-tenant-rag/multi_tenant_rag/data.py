"""Multi-tenant test corpus — three tenants with distinct document sets."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class TenantData:
    id: str
    name: str
    documents: List[str]


@dataclass(frozen=True)
class MultiTenantCorpus:
    tenants: List[TenantData]
    queries: Dict[str, List[str]] = field(default_factory=dict)
    # queries[tenant_id] = list of query strings
    relevant_doc_indices: Dict[str, Dict[int, List[int]]] = field(default_factory=dict)
    # relevant_doc_indices[tenant_id][query_idx] = list of doc indices
    # Doc indices are relative to the tenant's own document list


# ---------------------------------------------------------------------------
# Acme Corp — technology & engineering
# ---------------------------------------------------------------------------

ACME_DOCS = [
    "Acme Corp's cloud infrastructure runs on a hybrid multi-cloud architecture "
    "using AWS and Azure with Kubernetes orchestration for containerized workloads.",
    "Our data pipeline processes over 10TB of telemetry data daily using Apache "
    "Spark and Kafka for real-time streaming analytics.",
    "The engineering team follows a microservices architecture with API gateways "
    "and service mesh for inter-service communication.",
    "We use PostgreSQL for transactional data and Elasticsearch for full-text "
    "search across our product documentation.",
    "Security policies enforce zero-trust network access with VPC peering and "
    "IAM roles for all cross-account resource access.",
    "Our CI/CD pipeline uses GitHub Actions for automated testing and ArgoCD for "
    "GitOps-based deployments to Kubernetes clusters.",
    "The monitoring stack includes Prometheus for metrics collection and Grafana "
    "dashboards for real-time observability of production systems.",
    "Database backups run daily with point-in-time recovery and 30-day retention "
    "across all production PostgreSQL instances.",
]

ACME_QUERIES = [
    "cloud infrastructure and deployment",
    "data processing and analytics pipeline",
    "database and search infrastructure",
]

# For each query, indices into ACME_DOCS that are relevant
ACME_RELEVANT = {
    0: [0, 2, 5],  # cloud infra: hybrid cloud, microservices, CI/CD
    1: [1],         # data pipeline: Spark, Kafka
    2: [3, 7],      # database: PostgreSQL, backups
}

# ---------------------------------------------------------------------------
# GlobeBank — finance & banking
# ---------------------------------------------------------------------------

GLOBE_DOCS = [
    "GlobeBank's transaction processing system handles over 1 million daily "
    "transactions with 99.99% uptime across our core banking platform.",
    "Our fraud detection system uses machine learning models trained on historical "
    "transaction patterns to identify suspicious activity in real-time.",
    "Customer account data is encrypted at rest using AES-256 and in transit "
    "using TLS 1.3, with regular security audits by third-party firms.",
    "The mobile banking app supports biometric authentication, instant payments, "
    "and real-time balance notifications for retail customers.",
    "We comply with PCI-DSS, SOX, and GDPR regulations across all banking "
    "operations with quarterly compliance reporting.",
    "Our risk management team uses Value-at-Risk models and stress testing to "
    "assess portfolio exposure across different market conditions.",
    "The loan origination system automates credit scoring, document verification, "
    "and approval workflows for personal and business loans.",
    "Real-time payment processing integrates with FedNow, SWIFT, and SEPA "
    "networks for domestic and international money transfers.",
]

GLOBE_QUERIES = [
    "transaction processing and payments",
    "fraud detection and security",
    "compliance and regulations",
]

GLOBE_RELEVANT = {
    0: [0, 7],        # transactions: core banking, payments
    1: [1, 2],         # fraud + security
    2: [2, 4],         # compliance: encryption standards, regulations
}

# ---------------------------------------------------------------------------
# HealthPlus — healthcare
# ---------------------------------------------------------------------------

HEALTH_DOCS = [
    "HealthPlus manages electronic health records for 2 million patients across "
    "a network of 15 hospitals and 50 clinics in the northeastern United States.",
    "Our HIPAA-compliant platform encrypts all protected health information at "
    "rest and in transit with role-based access controls for healthcare providers.",
    "The patient scheduling system uses AI to optimize appointment booking, "
    "reducing wait times by 40% while maximizing provider utilization.",
    "Telemedicine services support video consultations, remote patient monitoring, "
    "and secure messaging between patients and healthcare providers.",
    "Our pharmacy management system integrates with major drug databases for "
    "real-time interaction checking and dosage verification.",
    "Medical imaging storage uses DICOM format with AI-assisted diagnosis "
    "tools for radiology and pathology departments.",
    "The billing and claims system processes insurance claims electronically "
    "with automated coding validation and denial management workflows.",
    "Population health analytics track disease prevalence, treatment outcomes, "
    "and public health metrics across our patient population.",
]

HEALTH_QUERIES = [
    "patient records and data security",
    "scheduling and telemedicine",
    "medical imaging and analytics",
]

HEALTH_RELEVANT = {
    0: [0, 1],         # records: EHR, HIPAA
    1: [2, 3],         # scheduling + telemedicine
    2: [5, 7],         # imaging + analytics
}


def default_corpus() -> MultiTenantCorpus:
    return MultiTenantCorpus(
        tenants=[
            TenantData(id="acme", name="Acme Corp", documents=ACME_DOCS),
            TenantData(id="globebank", name="GlobeBank", documents=GLOBE_DOCS),
            TenantData(id="healthplus", name="HealthPlus", documents=HEALTH_DOCS),
        ],
        queries={
            "acme": ACME_QUERIES,
            "globebank": GLOBE_QUERIES,
            "healthplus": HEALTH_QUERIES,
        },
        relevant_doc_indices={
            "acme": ACME_RELEVANT,
            "globebank": GLOBE_RELEVANT,
            "healthplus": HEALTH_RELEVANT,
        },
    )
