# Hiver SDE Intern Assignment — AppleSupport AI Agent

## 1. Problem Framing

**Brand chosen:** AppleSupport (from the Customer Support on Twitter dataset).
High volume, fairly consistent reply tone, and enough intent diversity
(account access, billing, software bugs, how-to questions) to build a real
taxonomy without being unmanageable at this scale.

**What "good" means for this agent:**
- **Correct intent classification** — routing a customer's actual problem to
  the right handling path.
- **Grounded reply drafting** — replies that reflect how AppleSupport has
  *actually* resolved similar issues historically, not generic LLM
  politeness with no connection to real brand behavior.
- **Conservative, explainable escalation** — the cost of wrongly
  auto-handling a billing dispute or an account-access issue is much higher
  than the cost of escalating a message that could have been auto-handled.
  The agent is deliberately biased toward escalating when uncertain.

**What I chose not to build, and why:**
- **Multi-turn conversation handling.** The dataset and this agent treat each
  customer message as a single-turn interaction (message -> reply). Real
  support threads often span several exchanges; handling that properly would
  require thread-level state, which was out of scope for the time available.
- **Actual DM-based resolution content.** A large fraction of AppleSupport's
  real replies are redirects to DM ("send us a DM and we'll continue there"),
  because anything involving account details moves off public Twitter. This
  agent's drafts mirror that same pattern where appropriate rather than
  trying to fabricate resolution details the brand itself keeps private.
- **Non-English support.** Several real customer messages in the dataset are
  in French, German, etc. The agent doesn't attempt to handle these -
  AppleSupport's own real replies to these cases are themselves just
  language-redirects, so this wasn't a meaningful gap to solve for v1.
- **Fine-tuning a custom classifier.** Used few-shot prompting against a
  general-purpose LLM instead of training/fine-tuning a dedicated model -
  appropriate given the taxonomy size (7 intents) and time available.
- **Multi-intent message splitting.** A message covering two issues (e.g. a
  bug report that also mentions a billing complaint) is classified into a
  single primary intent, not decomposed into multiple tickets.


## 2. Results vs. Baselines

Two baselines are used for comparison, evaluated against a 50-message sample
of the golden evaluation set (`eval/run_eval.py`):
- **Trivial baseline:** always predicts the majority-class intent from the
  golden set (`ambiguous_insufficient_context`), and the majority-class
  escalation decision (`auto`).
- **Simple baseline:** a keyword-matching heuristic classifier, using the
  same keyword rules originally built for golden-set sampling
  (`src/build_golden_set.py`), now reused as an actual classifier.

| Model    | Intent Accuracy | Escalation Accuracy |
|----------|-----------------|----------------------|
| Trivial  | 30.0%           | 60.0%                |
| Simple   | 50.0%           | 72.0%                |
| Pipeline | **84.0%**       | **94.0%**            |

The pipeline substantially outperforms both baselines on both metrics. The
gap over the simple keyword baseline (+34 points intent, +22 points
escalation) shows real value from using an LLM's language understanding
over pattern-matching, not just from having *some* automated system.

**Reply quality (LLM-as-judge, 1-5 scale, averaged over the same 50
replies):**

| Dimension    | Average Score |
|--------------|---------------|
| Helpfulness  | 3.14 / 5      |
| Tone match   | 3.98 / 5      |
| Grounding    | 4.28 / 5      |

Helpfulness lags noticeably behind tone and grounding - see Failure Mode 4
below, and note the grounding number's reliability is itself questioned in
Section 4.


## 3. Failure Analysis

**Failure mode 1 — high-confidence misclassification of ambiguity as
actionable.** During manual testing, the message "this keeps popping up too"
was classified as `ambiguous_insufficient_context` with **high** confidence,
but the original escalation rule only escalated on LOW confidence or
sensitive intents - it didn't account for the classifier being confidently
*certain that the message is ambiguous*. This let a genuinely unhandleable
message slip through as "auto." **Hypothesis:** the escalation logic
conflated "the classifier is unsure what this is" (low confidence) with
"the classifier is sure this doesn't contain enough information" (a
specific, confidently-assigned intent) - these needed separate handling.
Fixed by explicitly routing `ambiguous_insufficient_context` to always
escalate regardless of confidence.

**Failure mode 2 — genuine overlap between `post_update_performance_issue`
and `known_widespread_bug`.** The independent spot-check (see Section 4)
found multiple cases where a message like "frame rate stutters...
#MoreBugsOnTheOS" or a Wallet-app lock-screen regression could be
reasonably classified either way - a performance problem IS often a bug,
and the taxonomy doesn't cleanly separate "the phone runs badly" from "a
specific named glitch exists." **Hypothesis:** these two intents should
either be merged, or the taxonomy needs an explicit rule (e.g., "bug" only
applies when the customer names a specific, identifiable symptom, not a
general performance complaint) to be reliably distinguishable by a
classifier - it's currently ambiguous even to a careful human reader.

**Failure mode 3 — the LLM-judge has a leniency bias on grounding that
inflates the headline reply-quality number.** See Section 4 for full data:
the judge scored grounding at 5/5 on 11 of 15 spot-checked replies and never
below 3, while independent human scoring spread realistically across 3-4
and never reached 5. Example: for the reply "That's something we can
definitely dig into more. Let's continue in DM." (in response to "Yes, and
my phone dies at 40%.."), the judge scored grounding 4/5, while independent
review scored it 3/5 - a generic, low-content reply that doesn't reference
anything specific from the retrieved precedents. **Hypothesis:** the judge's
rubric prompt asks whether the reply "stays consistent" with a real support
reply, which a vague-but-harmless reply trivially satisfies; it doesn't
require the reply to actually *use* specific details from retrieval, so
genuinely well-grounded and merely-inoffensive replies score similarly.

**Failure mode 4 — templated, low-specificity replies for common complaint
types drag down helpfulness scores.** Helpfulness (3.14/5) is the lowest of
the three judge dimensions, and spot-check review confirms why: many drafted
replies default to a generic "let's continue in DM" / "send us more details"
pattern even when the customer's message already contains enough
information to give a concrete first troubleshooting step (e.g., "Yes, and
my phone dies at 40%.." scored helpfulness 2/5 independently - the reply
asks to move to DM without acknowledging the specific battery percentage
mentioned). **Hypothesis:** because a meaningful fraction of AppleSupport's
real historical replies (the retrieval precedents) are themselves
DM-redirects rather than actual troubleshooting content (see Problem
Framing), the model has learned to imitate this low-specificity pattern
even in cases where a real support agent might have given a more concrete
first step.

**Failure mode 5 — flat, generically apologetic tone on emotionally
charged/billing messages.** The lowest tone_match score in the spot-check
(2/5) was on a billing complaint: "y'all got me f***ed up charging my card
$12 and i only bought 3 songs and 2 ringtones.. I was only suppose to be
charged 7.74," which received the reply "Thanks for reaching out to us.
Please send us a DM with your country, and we'll look into this." The reply
doesn't acknowledge the specific dollar-amount discrepancy or validate the
customer's frustration before redirecting. **Hypothesis:** the drafting
prompt asks the model to match "tone and style" of retrieved precedents in
general, but doesn't explicitly instruct it to first acknowledge specific
emotionally-loaded details (a dollar figure, an accusation) before
redirecting - an easy, low-cost prompt improvement to test.


## 4. What Is Misleading About My Headline Number?

This section is mandatory, and honestly the most important one - here is
what I know that could make any single accuracy number look better than
it should:

- **Golden set labels are not fully independent of the model being
  evaluated.** Due to time constraints, ~185 of the 206 golden set rows were
  labeled using an AI-assisted pre-labeling workflow: the same classifier
  being evaluated generated a suggested intent, which was then reviewed and
  either accepted or corrected by hand. Only the first ~9 rows were labeled
  completely independently from scratch, plus ~11 rows that had no AI
  suggestion available and were labeled manually. **This means intent
  accuracy computed against the bulk of the golden set is partially
  circular** - rows where the human simply accepted the AI's suggestion
  will trivially "match" that same model's predictions later. The honest
  fix is to weight more heavily on the independently-labeled subset and the
  spot-check agreement rate (below) when interpreting the 84% headline
  number - real classification quality is probably somewhat lower than 84%.

- **Sample is temporally skewed toward one event.** Intent discovery
  (clustering a 600-message sample) showed roughly 54% of that sample was
  dominated by a single real-world incident - the iOS 11 launch backlash
  (battery drain, freezing, the autocorrect "I" bug). The golden set's
  intent distribution reflects this same skew, not necessarily
  AppleSupport's steady-state support volume across all time periods.

- **The retrieval index only covers 1,500 of 5,000 processed messages**, and
  uses a different embedding model (a local sentence-transformers model)
  than the one originally planned (Gemini's embedding API), switched
  midway due to persistent free-tier quota exhaustion. This may affect
  retrieval quality in ways not fully characterized.

- **The "simple" baseline has known, uncorrected weaknesses** (keyword
  collisions, e.g. "charged" vs. "icloud"; a short-message-length rule that
  can override otherwise-clear keyword matches) that make the pipeline look
  comparatively better than a more carefully-tuned simple baseline might.

- **The LLM-judge's grounding score (4.28/5 average) is likely inflated**
  and should not be treated as a precise or meaningful number as currently
  defined - see the agreement data immediately below.

**Independent spot-check of golden-set labels (20 rows, sampled separately
from the main golden set):** comparing one independent, from-scratch human
labeling pass against the existing (AI-assisted, bulk-templated) golden set
labels:

- **Intent agreement: 75%** (15/20)
- **Escalation agreement: 90%** (18/20)

This 75% is arguably a more honest estimate of "true" labeling reliability
than the headline number computed against the full golden set, since it's
not subject to the circularity described above. Reading through the 5
intent disagreements revealed three distinct patterns, not just noise:

1. **Genuine taxonomy boundary overlap** (see Failure Mode 2) - "Frame rate
   stutters... #MoreBugsOnTheOS" and the Wallet app boarding-pass bug were
   labeled `post_update_performance_issue` in the golden set but
   `known_widespread_bug` in the independent pass - both defensible.
2. **The bulk-template defaults missed nuance a careful read would catch.**
   "The issue is resolved temporarily after a restart, I'm having to
   restart every day" was labeled `ambiguous_insufficient_context` in the
   golden set, but independently reads as a clear `post_update_performance_issue`.
   Similarly, "It works now..." (a resolved/closing remark) was
   auto-escalated by the golden set's blanket rule, when `auto` is more
   sensible.
3. **At least one likely labeler error in the spot-check itself**, not the
   golden set: a message about music playback switching from Bluetooth to
   the charger was independently mislabeled as battery-related, likely a
   quick misread of the word "charger." This is an honest reminder that a
   single fast independent pass has its own noise.

**LLM-judge vs. human agreement (15-row spot-check on reply quality):**

| Dimension | Exact match | Within 1 point |
|---|---|---|
| Helpfulness | 46.7% | 100% |
| Tone match | 40.0% | 86.7% |
| Grounding | **0%** | **53.3%** |

Helpfulness and tone_match show reasonable agreement (100% and 87% within
one point respectively). **Grounding is a real problem**: the judge scored
grounding as 5/5 on 11 of 15 replies and never below 3, while independent
human scores spread more realistically across 3-4 and never reached 5. This
isn't random noise - it's a **systematic leniency bias**: the judge appears
to treat "doesn't say anything factually false" as sufficient for a high
grounding score, while a stricter human read of "grounding" - actually
engaging with specific details from the retrieved precedent, not just
avoiding contradiction - produced consistently lower scores. **Practical
implication: the reported average grounding score (4.28/5) is inflated and
should not be treated as a meaningful number as currently defined.**

**Practical takeaway:** treat the 84%/94% headline pipeline numbers as
directionally strong but not precise - real classification quality is
probably several points lower given ~25% intent disagreement even against
careful independent judgment, and the reply-quality grounding score should
be disregarded entirely until the judge's rubric is fixed (see below).


## 5. What I'd Do With One More Week

- Fully re-label the golden set independently (no AI-assisted pre-labeling),
  or at minimum expand the from-scratch spot-check to the full set to
  properly quantify how much the pre-labeling affected reported accuracy.
- Rewrite the LLM-judge's grounding rubric to be operationally specific
  (e.g., "does the reply reference concrete facts, steps, or context that
  appeared in the retrieved precedent?") rather than the current loose
  "stays consistent" framing, and re-run the human-agreement check to
  confirm it closes the gap.
- Improve the reply-drafting prompt to explicitly require acknowledging
  specific details from the customer's message (dollar amounts, named
  symptoms) before redirecting to DM, to address Failure Modes 4 and 5.
- Expand the retrieval index to the full 5,000-message pool.
- Handle multi-turn conversation threads rather than single message->reply
  pairs.
- Test generalization by running the same pipeline against a second brand
  (e.g. AmazonHelp) to see how much of the taxonomy and escalation logic is
  AppleSupport-specific vs. genuinely transferable.
- Add lightweight support for detecting non-English messages and routing
  them appropriately, rather than silently misclassifying them.