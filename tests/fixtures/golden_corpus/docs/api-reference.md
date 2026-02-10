# API Reference

Complete reference for the memory system public API.

## Core Operations

The episode method records structured experiences with objectives and outcomes.
The belief method creates or updates generalized knowledge statements.
The value method establishes core principles that guide behavior.
The note method captures quick factual observations and decisions.

## Search

Full-text search across all memory layers using keyword matching.
Semantic search available when an embedding model is configured.
Results are ranked by relevance score and recency.

## Processing

The process method runs model-driven memory promotion.
Exhaustion mode runs processing until convergence is reached.
Each transition can be triggered individually or as part of a full run.
