Known issues:

- Can't load timevol from external or network drives.
- On some machines, startup after recompiling can take 5-10 minutes. Cause unclear, possibly related to the on-board graphics chip (which is a pretty wild guess though).
- Loading settings before the .timevol causes all lines to be rendered with the background colour.

Syntax for preprocessor:

    Preprocessor.exe --json .\_datainfo.json --overwrite --channels velx:1,vely:1,velz:1,temp:1 -- outfilename

Individual fields from `_datainfo.json` can be overridden with explicit command-line flags.