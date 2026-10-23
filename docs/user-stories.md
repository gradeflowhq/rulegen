# `rulegen` User Stories

1. **Group related questions**  
   As an instructor, I want to explicitly group related questions, rename or ungroup them, and see what questions is in each group, so that shared grading logic (global rules/assumptions) is clear and controlled.

2. **Select authoring targets**  
   As an instructor, I want to select individual questions, grouped questions, or multiple independent questions, so that I can generate rules at the scope that matches my grading policy.

3. **Describe grading intent**  
   As an instructor, I want to provide my grading intension describing what should earn full, partial, or no credit, including required concepts, acceptable alternatives, and common errors, so that `rulegen` can interpret my grading policy.

4. **Provide targeted grading evidence**  
   As an instructor, I want to provide examples with different marks credited so that `rulegen` can use authoritative cases when testing a proposal. For grouped rules, I want to provide answer tuples spanning the selected questions; additional representative cases should be selected by the sampling system or LLM.

5. **Resolve missing context before generation** (LLM LEVEL)
   As an instructor, I want structured feedback when persisted question context, option text or required teacher evidence is missing, so that I can resolve the issue before generating a rule.

6.  **Generate constrained proposals**  
   As an instructor, I want `rulegen` to generate proposals using only rule families compatible with the selected target, so that the results are relevant, safe, and executable.

7.  **Review proposal evidence**  
    As an instructor, I want to see each proposal’s plain-language behaviour (LLM's interpretation of the grading intent), assumptions, expected versus actual results, coverage of representative submissions, feedback preview and diagnostics (especially if a failed proposal is presented when quota runs out), so that I can make an informed decision.

8.  **Refine or regenerate proposals** (Actually this is no different from the first interation, because rulegen is stateless)
    As an instructor, I want to add comments, labels, assumptions, or revised grading controls and rerun `rulegen`, or regenerate with the same request, so that I can improve or explore proposals while distinguishing my refinements from `rulegen`’s internal retries. 

9.  **Edit proposals manually**  
    As an instructor, I want to edit the generated rule directly, so that I can make precise changes while ensuring the edited rule is revalidated, retested, and reverified when its semantics change.

10. **Approve, apply, and grade explicitly**  
    As an instructor, I want to explicitly approve and apply a validated rule or reject it, then separately choose whether to grade all submissions, so that generated suggestions never change official rules or grades without my approval and final grading remains deterministic.

### Nice to Have (Phase 2)
1. **Set the proposal count limit** (Maybe this is not to be done on the instructor side, but rulegen needs to decide how mnay candidates to show)
   As an instructor, I want to set `k`, the maximum number of proposals Rulegen may generate, so that I can balance the range of alternatives against generation time and resource usage.

2. **Preserve proposal revisions** 
    As an instructor, I want an immutable, traceable revision history of request changes, selected baselines, returned proposals, test results, verification outcomes, and applied or rejected states, so that I can understand how the final rule was produced.

3. **Handle stale and partial results safely**  
    As an instructor, I want proposals affected by changed question content, context, or group membership to be marked stale, and failures for one target not to discard successful results for other targets, so that I can recover safely from changes and partial failures.

4. **Compare verified alternatives**  
    As an instructor, I want a recommended verified proposal and access to materially different verified alternatives, so that I can choose the rule that best matches my policy without seeing candidates that failed validation or verification.
