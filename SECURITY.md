# Security policy

Please report a suspected vulnerability through GitHub's private security
advisory interface for this repository. Do not open a public issue containing
credentials, private paths, exploit details, or sensitive dataset information.

The latest default-branch code is the supported development version. Research
checkpoints are Python-serialized artifacts: load only checkpoints produced by
a trusted RouteRec workspace, because compatibility loading may invoke pickle.
