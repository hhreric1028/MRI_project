# Archived pre-training attempt

This attempt was stopped after E1 completed epoch 1 because the ETA-only
`execution_started` monitoring field had not been initialized. The epoch itself
had finite losses and no AMP overflow, but no result from this attempt is used
for screening or comparison.

The complete suite logs, raw dual-channel datasets, fingerprints, preprocessed
datasets, and partial E1 output are retained here. The formal batch was restarted
from epoch 0 after initializing the monitoring field. No 1000-epoch run was
started.
