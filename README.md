# Distilling the Latent Space of a Real-Time Jet Tagger into Interpretable Physics Equations

The JEDI-linear jet tagger's last hidden layer, translated into if-statements on jet physics quantities.

Site: https://aaronw5.github.io/jet-tagger-distillation/ — one page per setup (all observables; no W/Z/top mass as a candidate threshold; no mass observables; no mass observables or exact equivalents; tuned for agreement) and per particle count (8, 64).

Control: the same first step on an untrained copy of the network explains a median 16% of each neuron's variance at 64 particles (55% at 8), against 92% (89%) for the trained network.
