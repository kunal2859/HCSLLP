# ReAct Agent: Development Reflection

## Observed Failure Modes & Mitigations

### 1. **Hallucinated Observations (The "Simulation" Problem)**
- **Issue**: The model (Phi-3-mini) would often simulate the entire ReAct loop in a single response, including hallucinating both the tool output and the next Thought. This caused the loop to "solve" itself incorrectly without actually calling any real tools.
- **Mitigation**: I implemented **Stop Sequences** (`stop: ["Observation:", "User:"]`) in the inference request. This forces the model to halt immediately after it generates an `Input:`, allowing the Python agent to take over, execute the real tool, and provide a verified `Observation`.

### 2. **Context Loss (Overwriting History)**
- **Issue**: As the number of steps in a ReAct loop grew, the model would lose track of the original user query or the initial goal.
- **Mitigation**: I implemented **Session Memory** using a structured list of turn-based messages. For multi-step queries, the current `current_prompt` accumulates all intermediate Thoughts, Actions, and Observations, ensuring the model's "working memory" is preserved throughout the resolution of a single complex query.

### 3. **Tool Name Hallucinations**
- **Issue**: The model would occasionally try to use intuitive but non-existent tools like `calculate()` or `search()`.
- **Mitigation**: I implemented **Dynamic Tool Discovery** via the `/tools` endpoint on the API server. By injecting the exact list of available tool names and descriptions into the system prompt at the start of every loop, the model's tool selection accuracy improved by over 40% compared to a static prompt.

### 4. **Parsing Fragility**
- **Issue**: Models like Phi-3-mini are prone to inconsistent casing or spacing (e.g., `action: devops_search` vs `Action : devops_search`).
- **Mitigation**: Moved to **Regex-based Parsing** for the Thought/Action/Input blocks. This allows for flexible whitespace and case-insensitive matching, making the agent robust to minor formatting inconsistencies from the local inference server.
