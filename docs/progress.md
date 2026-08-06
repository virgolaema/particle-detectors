# Progress

## Backlog
- Gamma notebooks: unify NIST XCOM data and attenuation helpers in library
- Neutrons notebook: move plotting/spectrum helpers to library
- Showers notebook: move materials + shower helpers to library
- Re-run all neutron notebooks against the corrected H/C cross sections
  (2026-08-06 fix, see projects/pns/NOTES.md) — the old H values were up to
  4x low in the epithermal region, so any conclusion that depends on
  moderation should be re-checked
- pns-3det-notes: full numbers pass after the MC fixes

## In Progress
- (none)

## Done
- PNS MC: gave the transport a clock (shield + tile), replaced the
  instantaneous single-pass tile model with a time-resolved 3-D random walk;
  predicted observable is now a die-away (lambda ~2.8 us vs 3.44 +/- 0.13 us
  measured). Fixed cubic-spline ringing in get_endf_cross_sections and added
  bound-atom H. Details in projects/pns/NOTES.md
- Bethe-Bloch notebook: plotting helpers moved to bethe_bloch_plotting.py
- Alpha notebook: materials and alpha helpers moved to alpha_lib.py, config moved to top
