# Fase D: testi proposti per il tracciamento

Regola 8 del piano: testo per `docs/findings.md` (in inglese) e righe per il log del documento di progetto (in
italiano, in `03c_righe_log_documento.txt`, controllate con lo script dei caratteri vietati). **Niente di questo è
stato scritto nella repository né nel documento online.** Aspetto il tuo ok, anche parziale.

---

## Sezione proposta per `docs/findings.md`

Da aggiungere dopo la sezione della parte online.

```markdown
## Confirmation and real latency (Phase D, 15 November 2026)

Scripts and logs in `outputs/fable_review/fase_d/` (not in the repository yet); notebook entries D.01 to D.04. The
methods were frozen and the predictions written before the confirmation data were opened.

**The 4 extra corruptions** (speckle noise, gaussian blur, spatter, saturate; logits exported for the 24 networks and
never used before). Over the last 10,000 inputs of the stream, the controller is within 0.006 of the oracle threshold
(-0.006 and -0.001 on CIFAR10 at severity 3 and 5, +0.004 and +0.005 on CIFAR100); the fixed clean threshold stays at
risk 0.078 and 0.140 on CIFAR10. The adaptation phase with the weighted fit gains +0.027, +0.041, +0.014 and +0.017
over the controller (95% intervals above zero in every cell), 13% to 17% of the drop, as much as the maps fitted with
access to the new domain. On the return to clean, the guard brings the long run risk back to that of the controller
alone (0.049 against 0.050 on CIFAR10, 0.048 against 0.049 on CIFAR100; 0.055 and 0.059 without the guard); it fires
in 79% of the runs on CIFAR10 and in all of them on CIFAR100.

**Natural shift: CIFAR-10.1** (Recht et al. 2018, 2,000 images, the 12 CIFAR10 networks). The accuracy of exit 9
falls by 9.5 points on average. The clean threshold reaches risk 0.066; certified savings with 1,000 calibration rows
are 74% of the clean ones (0.376 against 0.506). Online, the controller reaches 0.389 against 0.377 of the oracle
threshold. The logit maps recover nothing (+0.000 at the empirical oracle threshold, fitted on 1,000 images and
evaluated on the other 1,000): on this shift there is no systematic shift per class to correct.

**Real latency at batch 1** (one network per dataset, PyTorch eager). On CPU with one thread, time savings are 57% and
36% of the MAC savings on CIFAR10 (clean threshold on clean data, oracle threshold at severity 3), 14% on clean
CIFAR100, and negative on CIFAR100 under shift: the first three blocks work at 32x32 with 16 channels, are memory
bound, and take three quarters of the time for a third of the MACs. Reaching exit 9 with all heads costs 1.13 and
1.17 times the plain ResNet20. On MPS every exit adds about 0.4 ms (kernel launches and the synchronisation for the
comparison on the host) and early exit is always slower than the plain network.
```
