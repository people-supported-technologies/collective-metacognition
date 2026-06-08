# Attributed Conversational Causal Maps
## Proof of Concept Implementation Guide

## 1. Purpose

This proof of concept (POC) extracts **attributed conversational causal maps** from discussion transcripts.

The system should identify concepts mentioned by participants, extract causal or influence relationships between those concepts, preserve the provenance of each relationship, and visualise the resulting network at different levels of aggregation.

The intended output is not an objective causal model of the world. It is a structured representation of the causal beliefs, explanations, and mental models expressed by participants during a conversation.

For example, given the statement:

> “Recommendation algorithms feed you the information you want to see and you start believing vaccines cause autism.”

the system might extract:

```text
recommendation algorithms
    ── increase exposure to ──▶ congenial content

congenial content
    ── reinforces ──▶ belief proposition: “vaccines cause autism”

vaccines
    ── claimed to cause ──▶ autism

speaker_04
    ── expresses ──▶ belief proposition
```

Every extracted edge should link back to the original transcript evidence.

---

## 2. Core Product Idea

The system creates an inspectable graph of how people reason about an issue.

The same graph can be segmented and aggregated by:

- individual speaker;
- discussion table;
- room;
- discussion round;
- demographic group;
- participant cluster;
- whole event.

This enables comparisons such as:

- Which causal explanations recur across the whole discussion?
- Which relationships appear only within particular demographic groups?
- Where do groups agree on outcomes but disagree on mechanisms?
- Which edges are affirmed, rejected, or contested?
- How do causal explanations evolve across rounds of deliberation?

The longer-term aim is to support **collective metacognition**: giving participants and organisers a structured view of the reasoning emerging from the discussion.

---

## 3. Scope of the Initial POC

The first implementation should be deliberately narrow.

### In scope

Extract **explicit and near-explicit causal claims** from a transcript, preserve speaker attribution and evidence spans, and render an interactive graph.

Examples:

```text
“Social media increases political polarisation.”
“Cuts to youth services lead to more crime.”
“People lose trust when public bodies do not explain their decisions.”
```

### Out of scope for the first version

- inferring objective causal truth;
- deciding whether a participant is correct;
- generating policy recommendations;
- automatically merging all semantically similar concepts without review;
- reconstructing complete psychological models from unrestricted dialogue;
- feeding graph-derived recommendations back into live deliberation.

---

## 4. Design Principles

### 4.1 Preserve attribution

Every relationship must remain linked to:

- the original speaker;
- the table and discussion;
- the round;
- the transcript excerpt;
- the timestamp;
- the extraction method;
- a confidence score.

### 4.2 Separate claims from facts

The graph represents what participants said, not what the system asserts to be true.

For example:

```text
vaccines ── claimed to cause ──▶ autism
```

must be stored as an attributed proposition, not as an objective edge.

### 4.3 Prefer auditable extraction

The system should prioritise:

- transparent rules;
- typed relations;
- evidence spans;
- explicit confidence scores;
- human review for uncertain cases.

### 4.4 Aggregate late

Store atomic, attributed edges first. Aggregate only at query or visualisation time.

This makes it possible to compare speakers, tables, rounds, or demographic groups without losing provenance.

---

## 5. Conceptual Data Model

## 5.1 Node Types

Recommended initial node taxonomy:

| Node type | Description | Example |
|---|---|---|
| `actor` | Person, organisation, institution, or social group | `the government`, `the left`, `the media` |
| `technology` | Tool, platform, or technical mechanism | `recommendation algorithms` |
| `policy` | Law, programme, or intervention | `youth services`, `curfew rules` |
| `event` | Occurrence or change | `hospital closure`, `policy announcement` |
| `process` | Mechanism or ongoing activity | `exposure to repeated content` |
| `outcome` | Consequence or end state | `reduced trust`, `higher crime` |
| `value` | Normative concept | `fairness`, `freedom`, `safety` |
| `belief_proposition` | A claim or belief expressed in the discussion | `vaccines cause autism` |
| `topic` | Broad subject area | `immigration`, `public health` |

The taxonomy should remain configurable. The goal is not to create a perfect ontology in advance, but to impose enough structure for meaningful analysis.

## 5.2 Edge Types

Recommended initial relation ontology:

| Relation | Example |
|---|---|
| `causes` | `cuts to youth services → crime` |
| `increases` | `social media use → polarisation` |
| `reduces` | `community investment → crime` |
| `enables` | `training → employment` |
| `prevents` | `street lighting → antisocial behaviour` |
| `reinforces` | `repeated exposure → belief` |
| `undermines` | `lack of transparency → trust` |
| `increases_exposure_to` | `recommendation algorithm → congenial content` |
| `supports` | `local evidence → policy proposal` |
| `opposes` | `participant claim → policy proposal` |
| `associated_with` | Fallback relation where causality is unclear |
| `expresses` | `speaker → belief proposition` |
| `rejects` | `speaker → belief proposition` |
| `questions` | `speaker → belief proposition` |
| `reports` | `speaker → belief proposition` |

Use a small, controlled relation vocabulary initially. Add new relation types only when the existing ontology is clearly insufficient.

---

## 6. Input Schema

The POC should accept transcripts as JSON.

```json
{
  "discussion_id": "discussion_001",
  "room_id": "room_01",
  "table_id": "table_03",
  "round_id": "round_02",
  "turns": [
    {
      "turn_id": "turn_0001",
      "speaker_id": "speaker_04",
      "speaker_name": "Participant 4",
      "timestamp_start": "2026-06-08T10:30:00Z",
      "timestamp_end": "2026-06-08T10:30:09Z",
      "text": "Recommendation algorithms feed you the information you want to see and you start believing vaccines cause autism.",
      "demographics": {
        "age_group": "35-44",
        "location": "London"
      }
    }
  ]
}
```

Demographic attributes should be optional and access-controlled. Do not expose groups that are too small to report safely.

---

## 7. Output Schema

Each extracted edge should be stored as an atomic attributed record.

```json
{
  "edge_id": "edge_000145",
  "source_node": {
    "id": "concept_recommendation_algorithms",
    "label": "recommendation algorithms",
    "type": "technology"
  },
  "relation": "increases_exposure_to",
  "target_node": {
    "id": "concept_congenial_content",
    "label": "congenial content",
    "type": "process"
  },
  "speaker_id": "speaker_04",
  "discussion_id": "discussion_001",
  "room_id": "room_01",
  "table_id": "table_03",
  "round_id": "round_02",
  "turn_id": "turn_0001",
  "evidence_text": "Recommendation algorithms feed you the information you want to see",
  "stance": "asserted",
  "explicitness": "explicit",
  "confidence": 0.91,
  "extraction_method": "rule_and_llm_adjudication",
  "created_at": "2026-06-08T10:31:00Z"
}
```

Recommended values for `stance`:

```text
asserted
rejected
questioned
reported
uncertain
```

Recommended values for `explicitness`:

```text
explicit
near_explicit
inferred
```

For the first POC, display inferred edges separately and do not merge them silently with explicit edges.

---

## 8. Recommended Technical Architecture

## 8.1 High-Level Pipeline

```text
Transcript JSON
    ↓
turn and sentence segmentation
    ↓
concept extraction
    ├── standard NER
    ├── noun phrase extraction
    ├── domain rules
    └── zero-shot concept extraction
    ↓
coreference resolution
    ↓
candidate edge generation
    ├── dependency rules
    ├── OpenIE triples
    └── zero-shot relation extraction
    ↓
LLM adjudication for uncertain edges
    ↓
concept canonicalisation and deduplication
    ↓
atomic graph storage
    ↓
aggregation and graph queries
    ↓
interactive visualisation
```

## 8.2 Suggested Stack

### Backend

- Python 3.11+
- FastAPI
- Pydantic
- spaCy
- sentence-transformers
- NetworkX for the POC
- PostgreSQL or JSON files for the earliest prototype
- Neo4j when graph querying and segmentation become more complex

### NLP Components

- spaCy for:
  - sentence splitting;
  - tokenisation;
  - part-of-speech tagging;
  - dependency parsing;
  - standard NER;
  - noun phrase extraction;
  - custom rule-based extraction.
- GLiNER for flexible concept extraction using custom labels.
- `fastcoref` or another coreference-resolution model.
- OpenIE or dependency-pattern rules for candidate triple generation.
- GLiREL or a custom relation classifier for typed edge extraction.
- An LLM with schema-constrained JSON output for adjudicating ambiguous cases.

### Visualisation

- React frontend
- Ant Design for interface components
- AntV G6, Graphin, or Cytoscape.js for graph rendering
- Optional: Neo4j Bloom for internal exploration during prototyping

### Deployment

Start with an offline batch service. Once validated, deploy as a containerised microservice.

Suggested production shape:

```text
final transcript events
    ↓
message queue or Pub/Sub topic
    ↓
causal-map extraction service
    ↓
graph store
    ↓
analytics API
    ↓
React visualisation
```

---

## 9. Extraction Pipeline in Detail

## 9.1 Pre-processing

For each transcript:

1. split by speaker turn;
2. segment turns into sentences;
3. retain timestamps and speaker metadata;
4. resolve obvious transcription artefacts;
5. preserve the original raw text unchanged.

Do not overwrite the source transcript.

## 9.2 Concept Extraction

Classic NER alone is insufficient. It tends to identify named entities such as people, organisations, places, and dates, but misses broader concepts such as:

```text
the government
the media
the left
public trust
economic insecurity
recommendation algorithms
```

Use a combined approach.

### Layer A: Standard NER

Extract conventional entities:

```text
NHS
Facebook
Camden Council
China
```

### Layer B: Noun Phrase Extraction

Extract base noun phrases and candidate concepts:

```text
recommendation algorithms
public trust
the mainstream media
local businesses
youth services
```

### Layer C: Rule-Based Domain Ontology

Maintain configurable rules for recurring domain terms.

Example configuration:

```yaml
actors:
  - the government
  - the media
  - the left
  - the right
  - local councils
  - big tech

values:
  - fairness
  - trust
  - safety
  - freedom
  - accountability
```

### Layer D: Zero-Shot Concept Extraction

Use GLiNER or a comparable model with labels such as:

```text
actor
institution
political group
technology
policy
event
social issue
belief
value
mechanism
outcome
```

Store all extracted concept mentions before canonicalisation.

## 9.3 Coreference Resolution

Resolve references such as:

```text
they
it
these companies
the platform
that policy
```

Example:

```text
“Social media companies design these systems. They keep people engaged.”
```

should map:

```text
they → social media companies
```

Keep both the resolved and original surface forms.

## 9.4 Candidate Edge Generation

Use several complementary extraction methods.

### A. Dependency Rules

Create high-precision patterns for explicit causal language:

```text
causes
leads to
results in
increases
reduces
prevents
drives
makes people
because of
due to
as a result of
reinforces
undermines
```

Example:

```text
“Cuts to youth services increase crime.”
```

extracts:

```text
cuts to youth services ── increases ──▶ crime
```

### B. Open Information Extraction

Use OpenIE to generate additional subject–predicate–object candidates.

Example:

```text
“Recommendation algorithms feed people information they want to see.”
```

may yield:

```text
recommendation algorithms ── feed ──▶ information people want to see
```

### C. Zero-Shot Relation Extraction

Use GLiREL or a custom relation classifier to map candidate edges to the controlled ontology.

### D. LLM Adjudication

Use an LLM only for ambiguous or incomplete cases.

Do not ask the LLM to freely construct a graph from an entire transcript. Instead:

1. provide a short context window of one to three turns;
2. provide pre-extracted concept candidates;
3. restrict output to the approved relation ontology;
4. require a supporting quotation for every edge;
5. require a confidence score;
6. require `stance`;
7. require `explicitness`;
8. reject edges without transcript evidence.

Example prompt shape:

```text
Given the transcript excerpt and candidate concepts below, extract only relationships
supported by the text.

Allowed relations:
causes, increases, reduces, enables, prevents, reinforces, undermines,
increases_exposure_to, supports, opposes, associated_with.

For every edge, return:
source, relation, target, stance, explicitness, confidence, evidence_text.

Do not infer facts not stated or strongly implied in the excerpt.
```

---

## 10. Canonicalisation and Concept Merging

Different speakers may refer to similar concepts using different language:

```text
the authorities
central government
the government
Whitehall
```

Use sentence embeddings to propose candidate merges.

Recommended process:

1. embed each concept label and local context;
2. identify high-similarity concept pairs;
3. propose merges above a configurable threshold;
4. preserve original mentions;
5. allow manual approval or rejection;
6. maintain a canonical concept ID with aliases.

Do not automatically merge politically or semantically sensitive terms unless confidence is high.

Example:

```text
the media
social media
mainstream media
```

should not be collapsed automatically because participants may use them differently.

---

## 11. Aggregation Logic

Store atomic edges first.

At query time, aggregate by:

- speaker;
- table;
- room;
- round;
- participant cluster;
- demographic subgroup;
- whole event.

For every aggregated edge, calculate:

```text
number of mentions
number of unique speakers
number of tables
number of rounds
proportion asserted
proportion rejected
proportion questioned
mean confidence
share explicit vs inferred
first appearance
last appearance
```

Example aggregated output:

```json
{
  "source": "lack of transparency",
  "relation": "undermines",
  "target": "public trust",
  "unique_speakers": 22,
  "tables": 14,
  "rounds": 3,
  "stance_distribution": {
    "asserted": 0.82,
    "rejected": 0.05,
    "questioned": 0.13
  },
  "explicitness_distribution": {
    "explicit": 0.67,
    "near_explicit": 0.24,
    "inferred": 0.09
  }
}
```

---

## 12. Visualisation Requirements

The graph should be interactive and auditable.

### Core interactions

- filter by speaker, table, room, round, or demographic group;
- toggle between individual and aggregated graphs;
- click a node to inspect aliases and mentions;
- click an edge to inspect supporting transcript excerpts;
- filter edges by confidence;
- toggle inferred edges on or off;
- compare two groups side by side;
- inspect how the graph evolves across rounds.

### Visual encoding

Suggested conventions:

- edge thickness = number of unique speakers expressing the relationship;
- edge opacity = average confidence;
- solid line = explicit relationship;
- dashed line = near-explicit or inferred relationship;
- arrow direction = claimed direction of influence;
- edge label = relation type;
- node size = frequency or centrality;
- disagreement badge = relationship is asserted by some participants and rejected by others.

Avoid presenting contested edges as objective facts.

---

## 13. API Endpoints

Suggested FastAPI endpoints:

```text
POST   /extract
GET    /graphs/{discussion_id}
GET    /graphs/{discussion_id}/aggregate
GET    /graphs/{discussion_id}/compare
GET    /edges/{edge_id}
POST   /concepts/merge
POST   /concepts/unmerge
POST   /edges/{edge_id}/review
GET    /health
```

### Example extraction request

```json
{
  "discussion_id": "discussion_001",
  "turns": [...]
}
```

### Example graph query

```text
GET /graphs/discussion_001/aggregate?group_by=table_id&round_id=round_02
```

### Example comparison query

```text
GET /graphs/discussion_001/compare?group_a=location:Kilburn&group_b=location:Belsize
```

---

## 14. Evaluation Plan

Create a manually labelled evaluation set before optimising the pipeline.

### Recommended evaluation dataset

Start with:

- 200–500 transcript turns;
- a mixture of explicit and implicit causal claims;
- several topics;
- multiple speakers;
- examples containing negation, uncertainty, disagreement, and reported speech.

### Human Annotation Schema

For each turn, annotate:

```text
concept spans
canonical concept IDs
source node
target node
relation label
direction
stance
explicitness
supporting excerpt
```

### Evaluate Each Pipeline Stage

Measure precision, recall, and F1 for:

- concept extraction;
- concept normalisation;
- edge detection;
- relation classification;
- causal direction;
- stance classification;
- explicit vs inferred classification.

### Compare Pipelines

| Pipeline | Purpose |
|---|---|
| rules + dependency parsing | high-precision baseline |
| OpenIE | flexible candidate generation |
| GLiREL | typed relation extraction |
| constrained LLM | ambiguous and implicit cases |
| cascade | practical production candidate |

### Qualitative Review

Inspect:

- hallucinated edges;
- incorrect causal direction;
- missed negation;
- mistaken coreference;
- problematic merges;
- over-general concepts;
- politically sensitive misclassification.

---

## 15. Human Review Workflow

The POC should include a lightweight review interface.

Reviewers should be able to:

- approve or reject extracted edges;
- edit source or target labels;
- change relation types;
- change stance;
- correct causal direction;
- merge or split concepts;
- inspect original transcript evidence;
- flag sensitive or misleading representations.

Review feedback should be stored for later model improvement.

---

## 16. Privacy and Safety Requirements

Because the system processes conversational data, privacy must be designed in from the start.

### Minimum requirements

- retain speaker IDs rather than real names wherever possible;
- separate demographic metadata from raw transcript storage;
- restrict access to sensitive segmentation;
- suppress demographic slices below a minimum group size;
- retain evidence excerpts only where authorised;
- log graph queries and human edits;
- allow deletion of speaker-level data;
- encrypt data in transit and at rest.

### Interpretability requirement

Every displayed edge must be traceable to at least one transcript excerpt.

### Communication requirement

User-facing interfaces should clearly state:

> This graph represents relationships expressed or discussed by participants. It does not establish causal truth.

---

## 17. Implementation Phases

## Phase 1: Offline Notebook Prototype

Goal: prove extraction quality on a small transcript sample.

Deliverables:

- notebook or Python script;
- transcript loader;
- spaCy extraction;
- rule-based causal patterns;
- graph JSON output;
- NetworkX visualisation;
- manual evaluation on 50–100 turns.

## Phase 2: Hybrid Extraction Service

Goal: improve recall and structure.

Deliverables:

- GLiNER concept extraction;
- coreference resolution;
- OpenIE candidate triples;
- GLiREL or relation-classifier layer;
- constrained LLM adjudication;
- FastAPI endpoint;
- evaluation set of 200–500 turns.

## Phase 3: Interactive Graph Explorer

Goal: make the output inspectable.

Deliverables:

- React graph interface;
- edge provenance panel;
- group and round filters;
- side-by-side graph comparison;
- human review workflow.

## Phase 4: Integration with Live Deliberation

Goal: process final transcript events continuously.

Deliverables:

- message queue integration;
- incremental extraction;
- concept merge proposals;
- graph updates after each round;
- latency and cost benchmarks;
- privacy review.

## Phase 5: Collective Metacognition Experiment

Goal: test whether graph feedback improves deliberation.

Possible research questions:

- Does showing participants the graph improve shared understanding?
- Does it help participants identify unresolved disagreement?
- Does it improve the quality of subsequent discussion?
- Does it lead to convergence without suppressing minority viewpoints?
- Does it increase perceived transparency and legitimacy?
- Does it create anchoring or over-steering effects?

---

## 18. Suggested Repository Structure

```text
attributed-causal-maps/
├── README.md
├── pyproject.toml
├── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   └── labelled/
├── notebooks/
│   ├── 01_extraction_baseline.ipynb
│   ├── 02_relation_extraction.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── api/
│   │   └── main.py
│   ├── schemas/
│   │   ├── transcript.py
│   │   └── graph.py
│   ├── preprocessing/
│   │   ├── segmentation.py
│   │   └── coreference.py
│   ├── concepts/
│   │   ├── ner.py
│   │   ├── noun_phrases.py
│   │   ├── gliner.py
│   │   └── canonicalisation.py
│   ├── relations/
│   │   ├── rules.py
│   │   ├── openie.py
│   │   ├── glirel.py
│   │   └── llm_adjudicator.py
│   ├── graph/
│   │   ├── builder.py
│   │   ├── aggregate.py
│   │   └── store.py
│   └── evaluation/
│       ├── metrics.py
│       └── review.py
├── tests/
│   ├── test_concepts.py
│   ├── test_relations.py
│   ├── test_negation.py
│   └── test_aggregation.py
└── frontend/
    ├── package.json
    └── src/
```

---

## 19. Acceptance Criteria for the Initial POC

The first useful prototype should:

1. ingest transcript JSON with speaker metadata;
2. extract broad concepts beyond standard named entities;
3. extract explicit causal relationships;
4. distinguish asserted, rejected, questioned, and reported relationships;
5. attach every edge to a supporting transcript excerpt;
6. store atomic edges with speaker, table, and round metadata;
7. aggregate graphs by speaker and table;
8. render an interactive network;
9. allow edge inspection and human correction;
10. report baseline extraction quality on a manually labelled sample.

---

## 20. Key Risks

| Risk | Mitigation |
|---|---|
| Hallucinated edges | require evidence spans and confidence thresholds |
| Confusing claims with facts | store stance and provenance explicitly |
| Incorrect causal direction | evaluate direction separately and allow review |
| Over-merging concepts | propose merges rather than applying them silently |
| Missing minority viewpoints | aggregate late and preserve atomic edges |
| Demographic re-identification | suppress small cells and enforce permissions |
| LLM cost and latency | use rules and smaller models before LLM adjudication |
| Over-steering live deliberation | keep the first POC offline and observational |

---

## 21. Recommended First Sprint

A first two-week sprint could produce:

### Week 1

- define transcript input format;
- create 100-turn labelled sample;
- implement spaCy NER, noun chunks, and causal dependency rules;
- create atomic edge JSON;
- render a basic NetworkX graph.

### Week 2

- add zero-shot concept extraction;
- add constrained LLM adjudication;
- compare rule-only vs hybrid extraction;
- create a simple Streamlit or React explorer;
- review results manually and identify failure modes.

The output of the sprint should be a short technical memo reporting:

- extraction accuracy;
- example graphs;
- main error categories;
- estimated compute cost;
- recommended next steps.

---

## 22. Longer-Term Vision

The immediate objective is an offline analytical tool. The longer-term opportunity is a live metacognitive layer for deliberation.

A mature system could:

- display the causal explanations emerging across a discussion;
- highlight where participants agree on outcomes but disagree on mechanisms;
- surface contested assumptions;
- show how different groups understand the same issue;
- make changes in collective reasoning visible across rounds;
- help facilitators identify productive areas for further discussion.

The AI should not decide which causal explanation is correct or recommend which solution the group should adopt. Its role is to provide an inspectable representation of the group's reasoning, allowing humans to deliberate more effectively.
