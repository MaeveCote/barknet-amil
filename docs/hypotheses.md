**Central hypotheses — BarkNet-AMIL**

**H1 — Instance quality rises with patch size.** Larger patches yield higher per-patch (Stage-1) classification accuracy, because each patch carries more discriminative bark structure. *Confirm:* monotonic patch-level accuracy vs patch size. *Disprove:* flat or non-monotonic. *Status: partially supported* (384 > 288 > 224 at Stage 1, per your report — needs the patch-level numbers tabulated).

**H2 — Learned aggregation helps most when instances are weak.** The AMIL-vs-voting gap shrinks as per-patch accuracy rises; attention-weighting adds value by down-weighting noisy patches, and there's less noise to manage when patches are individually strong. *Confirm:* AMIL−vote gap declines monotonically with patch size, significant at small patches, non-significant at large. *Disprove:* gap constant across sizes, or larger at big patches. *Status: supported at two points* (+0.40pp sig at 224, −0.27pp ns at 384) — the spectrum is what turns two points into a trend.

Or it may be that learned aggregation is more useful when patches cover less area, leading to patches containing less discriminant feature and yielding patches that are "generic". This needs to be verified using entropy of the attention distribution normalized to the bag size.

**H3 — Training only on image level (stage 2) yield wild overfitting.** training only on bag level yield less gradient updates i.e. less variance in the seen gradients resulting in high overfitting and memorizing of the data. This will be compared using the standard 224 2 stage training and a 224 1 stage training. 

**H4 — There is a patch-size sweet spot; "bigger is better" is false as a blanket claim.** The size maximizing image-level accuracy differs from the size maximizing AMIL's contribution — accuracy favors large patches, aggregation-value favors small — so no single size dominates and the choice is a deliberate tradeoff. *Confirm:* accuracy peaks at a different size than the AMIL−vote gap. *Disprove:* one size wins on both. *Status: emerging* — this is the synthesis of H1–H3 and likely your headline.

Moreover, we can specify in practice the best DPI for the patches. This is useful information in the industry.

**H5 — Patch-based MIL beats whole-image classification.** The bag-of-patches approach outperforms classifying the resized whole image (the single-instance degenerate case). *Confirm:* whole-image-→224 accuracy clearly below the best patch-MIL. *Disprove:* whole-image ≈ or > patch-MIL — which would undercut the method's premise and is itself a publishable (if deflating) deployment finding. *Status: untested* — the baseline you're adding. This will be compared to our best model.

**H6 — Learned aggregation beats hard voting, at the mechanism level.** Holding weights fixed, attention pooling beats majority voting over the same patch head (isolating aggregation from extra training). *Confirm:* amil vs amil_vote significant and positive, robustly across folds. *Disprove:* gap vanishes or reverses under CV. *Status: supported on one split at 224* (p=0.008), reverses at 384 (feeds H2) — **rests entirely on the 5-fold CV holding; single-split McNemar on ~40 discordant images is not enough to publish.**

**H7 — There is a bag size that no longer gives better results.** All the above results will be given on uncapped bag test results. We will experiment on the standard 224 patch size model and reduce the bag size at test time from uncapped to 1. Smaller bag size should reduce the accuracy, making the probability of a lucky / unlucky bag of patches to appear higher.

**Two cross-cutting caveats that belong in the paper, not just your notes:**

- **The bag-size confound (H1/H3):** patch size moves quality and bag size together. State it explicitly; the decoupling run is what lets you claim either mechanism separately.
- **Everything is cross5 fold validated** All the above numbers will be reported as mean and standard deviation to prevent a lucky test set draw. 