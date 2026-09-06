# Foundation: why this project starts with a simulator

The final system will coordinate several charging stations under grid and uncertainty constraints. Before adding forecasting or distributed negotiation, we need a deterministic simulation whose results we can inspect and test.

The first implementation uses synthetic EV sessions. They let us create exact situations such as two EVs competing for limited power and verify that the controller never exceeds a charger, station, or grid limit. Only after these invariants pass will we replay real ACN-Data sessions.

We use three baseline policies:

- **Uncontrolled:** every connected EV charges at its maximum rate. This shows the grid-stress problem.
- **First-come, first-served:** simple operational baseline that can be unfair to late or urgent arrivals.
- **Equal share:** fairer power division but not deadline-aware; it can miss urgent deadlines.

These policies are not the final solution. They give an honest reference point for the future fairness-first optimizer.
