# knowledge_distillation_tensorflow
Knowledge Distillation tested with tensorflow 2.1.

## Prerequisites

- tensorflow 2.1.0
- numpy

## Features

- A framework for testing knowledge distillation with custom loss
- Tenosrboard added

## Tested Datasets

- Cifar10

## TODO

- [x] Implement Hinton knowledge Distillation loss
- [x] Implement training with soft logits from teacher (need a lot of system memory)loaded all on memory
- [X] Tested on small VGG 4-block network
- [ ] Implement efficient data loading and soft logits loading
- [ ] Test more loss fucntions
- [ ] Test more witht different dataset
- [ ] Test more with different model architecture

## References

Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).
https://github.com/dsskim/knowledge_distillation_tensorflow2
