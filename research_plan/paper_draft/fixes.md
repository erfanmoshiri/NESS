3- 
Fix the following typo/issues:

Section
Typo or issue
Recommended correction
Front matter
“Type of Paper (Article, Review, Communications, etc.)” remains in the manuscript.
Replace with the actual article type.
Front matter
“Surname of First Author et al.” remains in every page header.
Insert the correct running header or use the target IEEE template.
Front matter
DOI, dates, volume, issue and page-number placeholders remain.
Remove or replace according to submission-template requirements.
Front matter
The PDF uses the Transactions on Graph Intelligence and Network Applications template, not an IEEE template.
Reformat for the selected IEEE journal before submission.
Front matter
The title in “How To Cite” differs from the manuscript title and includes an unnecessary comma before “and.”
Make the title identical everywhere.
Abstract
“which combination of auxiliary tasks and view generation benefits…”
“which combinations of auxiliary tasks and view-generation strategies benefit…”
Abstract
“small binary datasets that do not represent real-world information” is inaccurate and overly broad.
State that they do not represent the dense-feature regime studied here.
Abstract
“Evaluating by downstream node-classification accuracy … improves macro-F1” mixes accuracy and macro-F1.
“Evaluated using macro-F1 on downstream node classification…”
Abstract
“up to 9.8%” does not state whether the gain is relative or absolute.
State “9.8% relative” and provide the underlying scores.
Abstract
Keywords end with a semicolon.
Remove the final semicolon.
Section 1
“when a large fraction of nodes, say 80%” is informal.
“when, for example, 80% of nodes…”
Section 1
“a feature-based neighborhood self-supervision” is ungrammatical.
“a feature-based neighbourhood self-supervised objective.”
Section 1
The roadmap directs readers to experiments in Section 5, whereas most experiments are in Section 6.
Correct the cross-reference or merge Sections 5 and 6.
Section 1
Novelty over reference [14] is insufficiently specified.
Add a formal conference-to-journal extension table.
Section 2 opening
The text says SSL is discussed first and missing-information methods second; the actual order is reversed.
Rewrite the roadmap to match the section order.
Section 2.3.1
Heading capitalisation differs from other headings.
Use consistent title case.
Section 2.3.2
Claims reconstruction “gives no downstream gain,” contradicting Table 5.
Reconcile the prose and reported result.
Figure 1 / Section 2.5
“Tree breakdown of models discussed in related work section.”
“Taxonomy of methods reviewed in Section 2.”
Section 2.5
“illustrates an breakdown summary”
“illustrates a breakdown” or “presents a taxonomy.”
Section 3
SSL is defined as requiring no human labels, but label-related objectives are then included.
Separate self-supervised and semi-supervised objectives.
Section 3
“Target difficulty.” appears as an isolated run-in heading.
Number it as a fifth axis or integrate it into the preceding text.
Table 1
Abbreviations such as “PE,” “betw.” and “rel.” are undefined.
Define them in the caption or avoid abbreviations.
Table 1
Several taxonomy entries have no direct citation.
Cite the source method for each objective family.
Section 3
“Appendix 8” is not a valid appendix reference.
Replace with “Appendix A.”
Figure 2
“We first apply a propagation … then contrastive…” begins the second sentence with lowercase “then.”
“We first apply Feature Propagation to prefill missing nodes. We then optimise…”
Figure 2
Caption lacks a final period and “SS labels” is undefined in the diagram.
Use “self-supervised targets” and add punctuation.
Section 4.2
“feature- propagation” contains an erroneous space.
“feature-propagation.”
Section 4.4
Claims that several view-generation options are examined in Section 5, but no systematic view table is provided.
Add the experiment and correct the section reference.
Section 4.5
Main-text histogram loss averages over (V_o), while Appendix A uses (V'_o).
Use one mathematically consistent valid-node set.
Section 4.5
Setting only (w_{\mathrm{ssl}}=0) is described as self-supervision-free, although edge and contrastive losses remain.
Rename the ablation and define every active loss explicitly.
Section 4.6
Label histograms are listed as an example of targets depending on “global structure,” although they are local.
Replace with a genuinely global target or remove the example.
Section 5
Section 5 contains only an introductory paragraph before Section 6 begins.
Merge it with Section 6 or make it a proper experimental-design section.
Section 5
Transferability and scalability are attributed to “Sections 6.7 and 6.9”; transferability is Section 6.8.
Change to “Sections 6.8 and 6.9.”
Section 6 setup
“main comparison against other benchmarks”
“main comparison against other methods/baselines.”
Section 6 setup
“Section 5 reports the selection” is incorrect.
Refer to Section 6.7.
Section 6 setup
“Appendix 8.1” is malformed.
Refer to Appendix B.
Table 4
“(\rho), Best per rate in bold.” has incorrect punctuation and capitalisation.
“(\rho). The best result at each rate is shown in bold.”
Table 4
The stated 0.95 condition is absent.
Add the row or revise the protocol description.
Table 5
Displayed deltas do not consistently match displayed rounded scores.
Compute deltas from the displayed values or report more decimals.
Figure 3
“each curve from a run with that objective alone.)” has an unnecessary period inside the parenthesis.
“each curve is from a run using that objective alone).”
Figure 3
The curves appear to be from single runs.
Plot seed means with uncertainty bands.
Figure 4
The y-axis label is cramped and nearly vertical letter by letter.
Increase figure width or shorten the label.
Figures 4–5
Two side-by-side panels are assigned separate figure numbers but described as left/right.
Use one figure with subfigures (a) and (b), or separate them.
Figure 5
Caption ends with a semicolon.
Replace it with a period.
Section 6.4
“identify the property that explains this” is too strong for a five-point correlation.
“investigate one property associated with the observed gains.”
Section 6.6
“hist and its classification variant” refers to a variant that is not defined.
State the exact objectives actually used.
Section 6.6
The explanation invokes neighbour-feature averaging although the discussed objectives are label- and structure-derived.
Rewrite the mechanism or run the appropriate feature-objective experiment.
Table 6
“−0.000”
“0.000.”
Table 6
“within noise” is claimed without uncertainty values.
Report standard deviations or confidence intervals.
Section 6.7
“which a mean-based aggregator does not” is ambiguous because GraphSAGE also uses mean neighbour aggregation.
Contrast the separate self-transformation with GCN-style aggregation explicitly.
Figure 7
“peaks near the selected settings” is not strictly true for every plotted parameter.
State that the chosen settings are within a stable region.
Figure 7
Hyperparameters appear to be selected using a plot of test macro-F1.
State clearly that selection used validation performance only.
Table 7
Caption says results use three seeds but no variation is reported.
Add mean ± standard deviation.
Figure 8
Plot title and caption say “Accuracy,” but the y-axis is macro-F1.
Replace “Accuracy” with “macro-F1.”
Section 6.9
“no memory penalty” conflicts with the plotted memory differences.
Say “manageable absolute memory usage” and report exact overhead.
Section 6.9
Hardware and whether preprocessing is included in runtime are unspecified.
Report GPU, CPU, software versions and timing boundaries.
Section 7
“the conclusion that self-supervision task is inert…”
“the conclusion that the self-supervised task is ineffective…”
Section 8
“auxilary”
“auxiliary.”
Author Contributions
Publisher template instructions and “X.X.” placeholders remain.
Replace with the authors’ actual CRediT contributions.
Funding
“grant number needed?” remains.
Insert the grant identifier or state that no grant number applies.
AI declarations
AI use is disclosed twice, and “generation of … statements and figures” is broad and potentially concerning.
Consolidate the disclosure and specify exactly which figures/content involved AI assistance.
Appendix A
Says objectives were evaluated in Section 5.
Refer to Section 6.3.
Appendix A
Centroid loss divides by (
N_k(i)\cap V_o
Appendix A
Positive path pairs may be within (\ell) hops while negatives are at least three hops away; the sets overlap if (\ell\geq3).
Specify (\ell<3) or use non-overlapping thresholds.
Appendix A
(k), (\ell), (K), (w_{ij}), sample counts and head architectures are unspecified.
Add all values to Appendix B.
Appendix B
“8.1. Appendix B” is malformed numbering.
Use “Appendix B. Hyperparameters and Design Choices.”
Table 9
Described as the “full” configuration but omits masking rate, Barlow-Twins (\lambda), projector, cluster size, negative ratio and objective-sampling parameters.
Expand the table to include all reproduction-critical settings.
References
Reference style is not IEEE style.
Convert the entire bibliography to the target journal’s IEEE format.
Reference 5
“pp. 11–1” appears incomplete or malformed.
Verify the article/page range.
Reference 19
“Neural Information Processing Systems (NeurIPS 2020.” has an unmatched parenthesis.
Correct the venue and punctuation.
References 35 and 39
“others” is used instead of a proper author list.
Supply the complete IEEE-compliant author list.
Reference 42
Begins with “TODO.”
Complete the author and bibliographic information before submission.
