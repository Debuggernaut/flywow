# What this is, and what it is not

This repo is a **cartoon of a published wiring diagram**, wired up so a video
game framebuffer can tickle six buckets of fly motor neurons. It exists to
point at the underlying science, not to impersonate it.

## The science it points at

The adult male *Drosophila* central nervous system connectome (brain + ventral
nerve cord) reconstructed by the FlyEM project at HHMI Janelia, the Drosophila
Connectomics Group (Cambridge / MRC LMB), and Google Research:

- Berg et al., *Sexual dimorphism in the complete Drosophila male central
  nervous system connectome*, Cell (2026).
- Dataset: neuPrint `male-cns:v1.0` / https://male-cns.janelia.org/
- License of the connectome data: CC BY 4.0.

That reconstruction is a genuine map: neurons, synapses, cell types,
neurotransmitter predictions, optic-lobe column coordinates. None of that is
a joke.

## The cartoon on top

Everything this trainer does after loading those tables is a toy:

- A leaky-integrate-and-fire cell with one shared set of time constants is
  **not** a fly neuron. Real cells have compartments, receptors, neuromodulation,
  and a lot of biology that is not in the edge list.
- `weight` is a **synapse count**, times a **guessed sign** from a
  neurotransmitter prediction, times a **learned scalar**. It is not a
  measured conductance.
- There is **no central pattern generator**, no proprioception, no muscle, no
  body. A “leg button” here is just the mean spike rate of a motor-neuron pool
  (`vnc_motor` + `subclass` + `somaSide` +, for hind legs, `exitNerve == MetaLN`).
- Kenyon cells and the mushroom body are left alone on purpose. This is not
  learning in the fly sense.

If the demo looks clever, credit the connectome. If a specific button fires,
credit the label file and a handful of scalars — not a mind.

