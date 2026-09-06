# Safe-MARL endpoint-data completion amendment

The frozen Safe-MARL protocol selects ACN sessions by connection time in a half-open test window and retains each session's true physical departure. This is the documented rule because clipping a session at the boundary would create an artificial deadline.

During the first final MAPPO replay, one valid session that connected on 31 December 2019 remained physically connected until 02:30 UTC on 1 January 2020. The 2019-only NSRDB input consequently lacked observations for ten required 15-minute execution steps. This was an execution-data boundary error; it did not reveal a model result.

`caltech_2019_safe_marl_matched_endpoint_amendment.json` leaves all pre-test choices unchanged and adds the official 2020 NSRDB GOES-CONUS V4.0.0 file for the identical point, interval, and PV proxy. The evaluation consumes only the needed 00:00--02:15 UTC endpoint records. Both solar-file hashes are written to the result provenance.

The original protocol configuration and every frozen training checkpoint remain unchanged. The resulting evidence must be called an **endpoint-completed amendment**, not an unqualified preregistered test.
