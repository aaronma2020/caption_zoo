### Sotf attention

---

| super parameters                |               |
| ------------------------------- | ------------- |
| optimizer                       | Adam          |
| decoder learning rate           | 4e-4          |
| encoder fine tune learning rate | 1e-4          |
| CNN                             | vgg19 with bn |
| batch size                      | 64            |



| method   | BLEU1 | BLEU2 | BLEU3 | BLEU4 | METEOR | ROUGE_L | CIDEr | SPICE |
| -------- | ----- | ----- | ----- | ----- | ------ | ------- | ----- | ----- |
| soft att | 72.4  | 55.5  | 41.5  | 31.1  | 25.2   | 53.4    | 95.8  | 18.1  |

