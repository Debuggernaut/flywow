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

```
What the “fly” is actually doing
A fruit fly does not have a camera and six keyboard legs. What we ran is a cartoon of one published wiring diagram: the male central nervous system connectome from Janelia / Princeton (neuPrint male-cns:v1.0). Every cell and every synapse we use is a real reconstructed neuron and a real counted contact. The dynamics — how voltage leaks, when a cell spikes, how a Minecraft frame becomes “light” — are a toy.
Path 1 — light into the optic lobe
The game frame is letterboxed into a 256×256 patch, converted to brightness (luma), and sampled on the fly’s own hexagonal eye map (assignedOlHex1/2). Each hex site is tied to optic-lobe intrinsic cells (ol_intrinsic). Brighter pixels make those cells more likely to fire, as a Poisson tick at up to 80 Hz. That is not an ommatidium and not color vision. It is “this patch of screen is lit → these reconstructed cells get extra current.”
Path 2 — optic lobe → projection neurons
Those hex-driven cells already synapse onto visual projection neurons (visual_projection, VPNs). In the animal, VPNs are the cables that carry a compressed version of the visual world out of the optic lobe toward the rest of the brain. We did not invent those edges. We only allowed a learned scale factor, θ, on the existing contacts (weight ≥ 5, sign from the predicted neurotransmitter: ACh-like +, GABA/glutamate-like −).
Path 3 — projection neurons → descending neurons
VPNs synapse onto descending neurons (descending_neuron, DNs). DNs are the neck of the funnel: a few thousand cells whose axons leave the brain and run the length of the ventral nerve cord. If this hop stays silent, the legs never hear the image. On raw game frames, at the published global gain, that is exactly what we measured — DNs at 0 Hz. Training was “turn the existing VPN→DN (and a few OL→DN) knobs up or down so labeled screenshots become more likely to push current through the neck.” We did not add synapses.
Path 4 — descending neurons → ventral cord motor neurons
DNs already contact ventral-nerve-cord cells (vnc_intrinsic, vnc_motor). Motor neurons were grouped by which leg they belong to — left/right × front/mid/hind — from the reconstruction metadata (somaSide, subclass, exitNerve). Each of those six pools is wired to one button. When the pool’s spike rate in a short window crosses “on,” that button is treated as pressed.
So the simulated chain is:
pixels → hex-sampled optic-lobe cells → visual projection neurons → descending neurons → leg motor neurons → six keys
Three cuts, all real edges:

ol_intrinsic → visual_projection
visual_projection → descending_neuron
descending_neuron → vnc_motor / vnc_intrinsic

plus a thin skip (ol_intrinsic → descending_neuron, only 71 edges) that barely exists in the map.
What the math is
Each cell is a leaky-integrate-and-fire unit: voltage leaks toward rest, incoming spikes add a signed conductance, a threshold makes a spike, then a short refractory. Synaptic weights are synapse counts × a sign × the fly’s global GAIN × exp(θ) on the trained types. θ was fit so that labeled frames (this screenshot → these legs) produce higher scores on the matching pools. Sparsity and sign never move.
What this is not
It is not a walking circuit. Real fly walking is a cord central-pattern generator plus mechanosensory feedback we never touch. Kenyon cells and the mushroom body are in the graph and can spike; we did not train them. There is no muscle, no joint, no retina optics, no learning rule the fly uses. A white pause screen already warms a few mid/hind pools — that is the connectome plus a hot gain, not “the fly recognized a menu.”
The honest sentence: we asked a published fly wiring diagram which of six reconstructed legs should twitch when this picture hits a fake eye, and we only scaled the cables that were already there. That it changes buttons with the live image means current is crossing those four hops. It does not mean the animal would play the game.
```