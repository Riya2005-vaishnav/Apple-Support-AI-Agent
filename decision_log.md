# Decision Log

Non-obvious decisions made while building this project, and why.

1. **Chose AppleSupport as the brand.** High volume in the dataset, consistent
   reply tone, and a clear mix of intent types (account/billing/bugs/how-to) -
   good balance of enough data without being unmanageable.

2. **Subsampled to 5,000 customer->brand pairs with a fixed random seed
   (random_state=42).** Keeps the pipeline reproducible in under 15 minutes,
   as required, while still giving enough volume for a real retrieval index
   and golden set.

3. **Reconstructed pairs only when the parent tweet was `inbound=True`
   (genuinely from a customer).** Brand-to-brand follow-up tweets in the same
   thread were dropped, since they're not real customer-originated pairs.

4. **Discovered the intent taxonomy via embedding + clustering, not by
   guessing categories upfront.** Revealed that ~54% of a random sample was
   dominated by a single real-world event (the iOS 11 launch backlash) -
   this directly shaped the "misleading headline number" analysis.

5. **Split "software update issue" into two separate intents (performance
   vs. battery drain) instead of merging them**, since they have different
   urgency/severity profiles even though they co-occur often.

6. **Added `ambiguous_insufficient_context` as its own intent**, rather than
   forcing short/vague follow-up messages into one of the "real" categories.
   This also became a rule in the escalation logic (see #13).

7. **Used Gemini (`gemini-embedding-001` then switched to a local model, see
   #10; `gemini-3.6-flash` for generation) rather than a self-hosted LLM**,
   for simplicity and to avoid needing local GPU infrastructure.

8. **Built retrieval with plain numpy + cosine similarity instead of a vector
   database (FAISS, etc.).** At ~1,500 vectors, brute-force search is
   millisecond-fast; a dedicated vector DB only matters at much larger scale,
   and this keeps the code simple enough to fully explain line-by-line.

9. **Sampled the retrieval index at 1,500 messages (not the full 5,000).**
   Balanced index coverage against embedding cost/time.

10. **Switched retrieval embeddings from Gemini's API to a local
    sentence-transformers model (all-MiniLM-L6-v2) after repeatedly hitting
    Gemini's restrictive free-tier embedding quota**, which kept blocking
    progress across multiple sessions. Kept Gemini for classification/
    generation, where the quota was more workable. This removes rate limits
    entirely for retrieval, which is called on every single message at
    runtime (not just once).

11. **Used AI-assisted pre-labeling for the golden set (not fully independent
    hand-labeling of all 206 rows)**, due to time constraints - the
    classifier's own suggestions were used as a starting draft, reviewed/
    corrected by hand, with the first ~9 examples labeled fully independently
    and ~11 unsuggested rows fixed manually. This is a real tradeoff in
    rigor vs. speed; the honest spot-check comparing independent labels to
    the AI's suggestions is reported separately (see report).

12. **Batched LLM classification calls (multiple messages per API request)
    instead of one call per message**, after discovering Gemini's free tier
    limits by REQUEST COUNT, not content volume - this cut the number of
    calls needed by ~15x for pre-labeling.

13. **Escalation logic is simple, explainable if/else rules (not another LLM
    call)**: escalate if the intent is billing/account-access/ambiguous, OR
    classifier confidence isn't "high", OR no sufficiently similar past case
    exists. Chose explainability and speed over a fancier learned escalation
    model.

14. **Found and fixed a real escalation logic bug during manual testing**: a
    high-confidence `ambiguous_insufficient_context` classification was
    slipping through as "auto" because the original rule only escalated
    sensitive intents or LOW confidence - it didn't account for a message
    being confidently classified AS ambiguous. Fixed by explicitly routing
    that intent to always escalate.

15. **Kept the "simple" baseline deliberately simple (keyword matching)**,
    even after noticing it makes real classification errors (e.g. "charged"
    vs. "icloud" keyword collision, or short-message rule overriding a clear
    how-to question). These are legitimate baseline weaknesses, useful for
    showing why the LLM pipeline outperforms a rule-based approach - not
    bugs to be fixed.