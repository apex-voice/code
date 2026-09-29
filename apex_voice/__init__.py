"""APEX-Voice: measuring professional work completion by full-duplex voice agents.

Subpackages:

- ``schemas``      -- pydantic v2 data models for tasks, users, tools, artifacts, grading, run logs.
- ``config``       -- typed run configuration, seeds, and version stamps.
- ``environments`` -- state store, tool runtime, commit guard, external events, knowledge base.
- ``artifacts``    -- versioned workspace, field graders, artifact graders, consistency checks.
- ``user_sim``     -- deterministic user simulator: hidden state, flow engine, act observer,
                      frozen realization bank, and speech synthesis.
- ``harness``      -- run loggers, clocks, audio utilities, the text runner, and the full-duplex
                      realtime runner.
- ``adapters``     -- the reference text agents and the realtime speech-to-speech model adapters.
- ``scoring``      -- Production Task Score (PTS), predicates, the semantic field judge,
                      duplex and latency metrics.
- ``analysis``     -- aggregation of campaign runs into the benchmark result tables.
- ``tasks``        -- task loading, validation, and taxonomy linting.
"""

__version__ = "1.0.0"
# Benchmark specification version. Bump on any change that alters task semantics or grading.
BENCHMARK_VERSION = "v1.0"
