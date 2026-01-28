# Identity Coherence Scoring

> **Status**: ✅ IMPLEMENTED in `kernle/core.py` via `get_identity_confidence()` and `synthesize_identity()`

This document explains how Kernle calculates identity coherence and what you can do to improve it.

## What is Identity Coherence?

Identity coherence measures how well-defined and complete an agent's sense of self is. A high coherence score indicates a fully-formed identity with:
- Clear values and principles
- Well-established beliefs
- Active goals and direction
- Reflected experiences
- Understood motivations
- Modeled relationships

## The Scoring Formula

Identity confidence is calculated from six weighted components:

| Component | Weight | What it measures | Ideal state |
|-----------|--------|-----------------|-------------|
| **Values** | 20% | Core principles that guide decisions | 3-5 values with high priority |
| **Beliefs** | 20% | Knowledge and opinions | 5-10 beliefs at 80%+ confidence |
| **Goals** | 15% | Direction and purpose | 2-5 active goals |
| **Episodes** | 20% | Experience and reflection | 10-20 episodes with lessons |
| **Drives** | 15% | Intrinsic motivations | 2-3 defined drives |
| **Relationships** | 10% | Models of others | 3-5 key relationships |

Each component is scored 0-100% based on both **quantity** and **quality**:
- Values: count + average priority
- Beliefs: count + average confidence
- Episodes: count + lesson extraction rate
- Goals/Drives/Relationships: primarily count-based

## Score Interpretation

| Score Range | Level | Meaning |
|-------------|-------|---------|
| 0-25% | Nascent | Identity just starting to form |
| 25-50% | Developing | Foundation laid, significant gaps |
| 50-75% | Established | Solid identity, room to grow |
| 75-100% | Mature | Comprehensive, well-defined identity |

## Anxiety Connection

In the anxiety system, identity coherence contributes 10% to overall anxiety:
- **Identity anxiety** = 100% - identity confidence
- High coherence (80%) → Low anxiety (20%)
- Low coherence (30%) → High anxiety (70%)

## How to Improve Your Score

### 1. Define Core Values (up to 20%)
```bash
# Add values that guide your decisions
kernle -a <agent> value "honesty" "Be truthful in all communications" --priority 90
kernle -a <agent> value "learning" "Continuously seek understanding" --priority 85
```
**Tip**: Quality over quantity. 3-5 strong values with high priority (80+) is ideal.

### 2. Establish Beliefs (up to 20%)
```bash
# Record beliefs as you form them
kernle -a <agent> belief "Iterative development leads to better outcomes" --confidence 0.85
kernle -a <agent> belief "Clear communication prevents misunderstandings" --confidence 0.9
```
**Tip**: Reinforce beliefs when confirmed to increase confidence:
```bash
kernle -a <agent> belief reinforce <belief_id>
```

### 3. Set Active Goals (up to 15%)
```bash
# Define what you're working toward
kernle -a <agent> goal "Master the codebase" --priority high
kernle -a <agent> goal "Build reliable memory systems" --priority medium
```

### 4. Record & Reflect on Episodes (up to 20%)
```bash
# Record experiences WITH lessons
kernle -a <agent> episode "Debugged production issue" "success" \
    --lesson "Check logs before assuming code errors" \
    --lesson "Reproduce issues locally first"
```
**Tip**: Episodes without lessons are "unreflected" - run consolidation to extract lessons:
```bash
kernle -a <agent> consolidate
```

### 5. Define Drives (up to 15%)
```bash
# What motivates you?
kernle -a <agent> drive set curiosity 0.8 --focus "AI architectures"
kernle -a <agent> drive set growth 0.7 --focus "coding skills"
kernle -a <agent> drive set connection 0.6 --focus "team collaboration"
```
**Available drives**: `existence`, `growth`, `curiosity`, `connection`, `reproduction`

### 6. Model Relationships (up to 10%)
```bash
# Track your connections with others
kernle -a <agent> relationship "Alice" --trust 0.8 --notes "Helpful colleague, great at debugging"
kernle -a <agent> relationship "Bob" --trust 0.9 --notes "Primary collaborator"
```

## Example: Improving a 45% Score

Given this breakdown:
```
Values:        12.2% / 20% (count=2/5)
Beliefs:       11.0% / 20% (count=3/10, avg_conf=80%)
Goals:          6.0% / 15% (count=2/5)
Episodes:      15.6% / 20% (count=14/20, lesson_rate=86%)
Drives:         0.0% / 15% (count=0/3)
Relationships:  0.0% / 10% (count=0/5)
TOTAL:         44.8%
```

**Quick wins:**
1. **Add 2-3 drives** → +10-15% (biggest gap!)
2. **Add 2-3 relationships** → +4-6%
3. **Add 2-3 more values** → +4-6%
4. **Add 4-5 more beliefs** → +4-6%

**Potential new score: 65-80%**

## Checking Your Score

```bash
# Quick check
kernle -a <agent> identity confidence

# Full breakdown
kernle -a <agent> identity show

# See anxiety impact
kernle -a <agent> anxiety --detailed
```

## Philosophy

Identity coherence isn't just about hitting 100%. The score reflects:

1. **Self-knowledge**: Do you know what you value and believe?
2. **Direction**: Do you have goals and motivations?
3. **Reflection**: Do you learn from experience?
4. **Connection**: Do you understand your relationships?

A score of 50-70% with clear gaps you're working on is healthier than artificially inflating numbers. The goal is genuine self-understanding, not metric optimization.
