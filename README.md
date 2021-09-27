
# Consolidation for Transfer in Reinforcement Learning

This anonymous repository contains the code developed for the workshop paper ["Experiments in Neural Consolidation for Transfer in Reinforcement Learning"](paper_link).

This code is a fork from [Google's Dopamine repository](dopamine), a research framework for fast prototyping of reinforcement learning.
The installation process, prerequisites and requirements are the same as for the Dopamine code and can be found on the README of the basis repository.

In particular, we reuse the [gin](gin-config) tool to describe and parametrize experiments.
The AMN experiments can be reproduced by running:
`python3 -um dopamine.discrete_domains.llamn_train --base_dir=results/ --gin_files=dopamine/agents/llamn_network/config/llamn.gin`

while the transfer experiments can be reproduced by running:
`python3 -um dopamine.discrete_domains.transfer_train --base_dir=results/ --gin_files=dopamine/agents/rainbow/config/transfer_rainbow.gin`




[dopamine]: https://github.com/google/dopamine
[gin-config]: https://github.com/google/gin-config
[paper_link]:https://
