# Rule: GitHub Task Lifecycle & Documentation Standards

Always follow these guidelines when managing tasks, issues, and project boards:

## 1. Task State Transitions
* **Active Work**: When starting a task, immediately move the corresponding Project Board item to the `In Progress` column. Do not leave it in `Todo`.
* **Done Column**: Do NOT automatically move a task to the `Done` column upon drafting or committing the code. A task must remain in `In Progress` until the user explicitly reviews the output and says "validate and ok" (or similar confirmation).

## 2. Completion Comments
* **Required Documentation**: When moving a project item to `Done`, you must add a comment to the corresponding GitHub Issue detailing the completion.
* **Comment Template**:
  ### Task Completed
  * **Deliverables**: [Summary of what was built/written]
  * **Created/Modified Files**:
    * `path/to/file` (brief purpose)
  * **Git Release**: Tagged under version `X.Y.Z` and pushed to origin.
  * **Validation/KPIs**: [Metrics, schema checks, or tests run]

## 3. Projects V2 Mutations (Model Knowledge)
* **Draft Issues**: You cannot edit single-select fields or assignees on Draft Issues. You must first convert them to real issues using `convertProjectV2DraftIssueItemToIssue` before applying labels/assignees.
* **Mutations**: Use GraphQL `updateProjectV2ItemFieldValue` for column/status updates and `addComment` for completion comments.
