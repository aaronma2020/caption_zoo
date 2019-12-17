### show attention and tell(NIC)

---

**Difference from paper**:

| method        | we   | paper      |
| ------------- | ---- | ---------- |
| optimizer     | Adam | SGD        |
| learning rate | 4e-4 | Don't know |
| batch size    | 80   | Don't know |



No fine tune CNN:

| method | BLEU1 | BLEU2 | BLEU3 | BLEU4 | METEOR | ROUGE_L | CIDEr | SPICE |
| ------ | ----- | ----- | ----- | ----- | ------ | ------- | ----- | ----- |
| nic    | 66.8  | 49.1  | 35.5  | 26.2  | 22.6   | 49.4    | 79.0  | 15.5  |



Fine tune CNN:

| method | BLEU1 | BLEU2 | BLEU3 | BLEU4 | METEOR | ROUGE_L | CIDEr | SPICE |
| ------ | ----- | ----- | ----- | ----- | ------ | ------- | ----- | ----- |
| nic    | 71.2  | 54.2  | 40.7  | 30.8  | 25.3   | 53.0    | 95.6  | 17.7  |

