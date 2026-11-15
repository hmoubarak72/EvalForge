"""Composition root: builds and wires every service exactly once.

All dependencies (settings, engine, clock, model clients) are passed in
explicitly, so tests and the demo can substitute doubles without patching
globals. The API, CLI, scheduler and dashboard all start from
:func:`build_container`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from evalforge.adapters.llm import (
    ChatClient,
    EmbeddingClient,
    build_chat_client,
    build_embedding_client,
    build_target_client,
)
from evalforge.adapters.sources import supported_kinds
from evalforge.config import Settings, get_settings
from evalforge.db.base import utcnow
from evalforge.db.models import IngestionSource
from evalforge.db.session import make_engine, make_session_factory
from evalforge.domain.errors import InvalidRequestError
from evalforge.domain.schemas import EndpointConfig
from evalforge.pipeline.orchestrator import PipelineOrchestrator
from evalforge.repositories import InteractionRepository
from evalforge.services.classification import (
    ClassificationService,
    HeuristicQualityJudge,
    LLMQualityJudge,
    QualityJudge,
)
from evalforge.services.clustering import (
    ClusterNamer,
    ClusterService,
    DeterministicClusterNamer,
    LLMClusterNamer,
)
from evalforge.services.dedup import DedupService
from evalforge.services.evaluation import EvaluationService
from evalforge.services.ingestion import IngestionService
from evalforge.services.labelers import HeuristicLabeler, Labeler, LLMLabeler
from evalforge.services.labeling import LabelingService
from evalforge.services.metrics import DatasetMetricsService
from evalforge.services.redaction import Redactor
from evalforge.services.regression import RegressionService
from evalforge.services.review import ReviewService
from evalforge.services.sampling import SamplingService
from evalforge.services.scoring import LLMGrader, Scorer

Clock = Callable[[], datetime]


@dataclass
class Container:
    """Wired application services."""

    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    clock: Clock
    embedder: EmbeddingClient
    redactor: Redactor
    ingestion: IngestionService
    sampling: SamplingService
    classification: ClassificationService
    labeling: LabelingService
    dedup: DedupService
    review: ReviewService
    metrics: DatasetMetricsService
    regression: RegressionService
    evaluation: EvaluationService
    orchestrator: PipelineOrchestrator


def build_container(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    clock: Clock = utcnow,
    embedder: EmbeddingClient | None = None,
    judge_chat: ChatClient | None = None,
    labeler_chat: ChatClient | None = None,
    target_client_factory: Callable[[EndpointConfig], ChatClient] | None = None,
) -> Container:
    """Build the service graph.

    In offline mode (the default) every model role uses deterministic local
    implementations. Passing ``judge_chat`` / ``labeler_chat`` explicitly, or
    disabling offline mode with an ``openai_compatible`` provider, switches the
    corresponding role to LLM-backed implementations with heuristic fallback.
    """
    settings = settings or get_settings()
    engine = engine or make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    embedder = embedder or build_embedding_client(settings)
    judge_chat = judge_chat or build_chat_client(settings, settings.judge)
    labeler_chat = labeler_chat or build_chat_client(settings, settings.labeling)

    heuristic_judge = HeuristicQualityJudge(settings.classification)
    judge: QualityJudge = (
        LLMQualityJudge(judge_chat, heuristic_judge) if judge_chat else heuristic_judge
    )
    namer: ClusterNamer = LLMClusterNamer(judge_chat) if judge_chat else DeterministicClusterNamer()
    heuristic_labeler = HeuristicLabeler(seed=settings.labeling.seed)
    labeler: Labeler = (
        LLMLabeler(labeler_chat, heuristic_labeler, settings.labeling.pass_temperature)
        if labeler_chat
        else heuristic_labeler
    )

    def scorer_factory(mode: str) -> Scorer:
        if mode in ("llm", "hybrid"):
            if judge_chat is None:
                raise InvalidRequestError(
                    f"grading mode '{mode}' needs a judge model; configure EVALFORGE_JUDGE__* "
                    "and disable EVALFORGE_OFFLINE_MODE, or use 'deterministic'"
                )
            return Scorer(mode, settings.evaluation.pass_score_tolerance, LLMGrader(judge_chat))
        return Scorer("deterministic", settings.evaluation.pass_score_tolerance)

    redactor = Redactor(settings.redaction)
    metrics = DatasetMetricsService(clock)
    ingestion = IngestionService(redactor, clock)
    sampling = SamplingService(settings.sampling, clock)
    classification = ClassificationService(
        settings.classification,
        embedder,
        ClusterService(settings.clustering, namer),
        judge,
        clock=clock,
    )
    labeling = LabelingService(settings.labeling, settings.dedup, labeler, metrics, clock)
    dedup = DedupService(settings.dedup, settings.labeling, labeler.model_name, clock)
    regression = RegressionService(settings.regression)
    evaluation = EvaluationService(
        settings.evaluation,
        scorer_factory,
        target_client_factory or build_target_client,
        regression,
        clock,
    )
    orchestrator = PipelineOrchestrator(
        session_factory, ingestion, sampling, classification, labeling, dedup, metrics, clock
    )
    return Container(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        clock=clock,
        embedder=embedder,
        redactor=redactor,
        ingestion=ingestion,
        sampling=sampling,
        classification=classification,
        labeling=labeling,
        dedup=dedup,
        review=ReviewService(clock),
        metrics=metrics,
        regression=regression,
        evaluation=evaluation,
        orchestrator=orchestrator,
    )


def register_source(
    session: Session, name: str, kind: str, config: dict[str, object], enabled: bool = True
) -> IngestionSource:
    """Create or update a log source by name."""
    if kind not in supported_kinds():
        raise InvalidRequestError(
            f"unknown source kind '{kind}'; supported: {', '.join(supported_kinds())}"
        )
    source = InteractionRepository(session).get_source_by_name(name)
    if source is None:
        source = IngestionSource(name=name, kind=kind, config=dict(config), enabled=enabled)
        session.add(source)
    else:
        source.kind, source.config, source.enabled = kind, dict(config), enabled
    session.flush()
    return source


def sync_configured_sources(session: Session, settings: Settings) -> int:
    """Upsert the sources declared in ``EVALFORGE_SOURCES`` into the database."""
    for source in settings.sources:
        register_source(session, source.name, source.kind, source.config, source.enabled)
    return len(settings.sources)
