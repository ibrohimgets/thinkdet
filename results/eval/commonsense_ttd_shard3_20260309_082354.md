# Think-then-Detect Commonsense Results

- n_samples: 98
- shard: 3/8

## Overall

| condition | hit@0.5_top1 | hit@0.5_topk | mean_iou_top1 |
|---|---:|---:|---:|
| baseline_direct | 0.3776 | 0.6531 | 0.3660 |
| baseline_translated | 0.6224 | 0.7653 | 0.5764 |
| thinkdet_direct | 0.4082 | 0.6735 | 0.3962 |
| thinkdet_translated | 0.6020 | 0.7551 | 0.5620 |

## Sample Translations

| object | original prompt | translated |
|---|---|---|
| person | someone you ask questions . | cell phone |
| chair | seat for one person . | chair |
| chair | something used while eating . | tennis racket |
| giraffe | wild animal kids like watching . | giraffe |
| zebra | striped animal people see at zoos . | zebra |
| elephant | wild animal behind a fence . | elephant |
| bowl | round dish for food . | bowl |
| bowl | something used at meals . | fork |
| couch | place to rest indoors . | couch |
| pizza | food people eat hot . | pizza |
| horse | animal used for riding lessons . | dog |
| sandwich | quick meal between bread . | sandwich |
| sandwich | food made with fillings . | hot dog |
| car | ride used on roads . | bus |
| dining table | furniture where meals are served . | table |
| motorcycle | something worn with a helmet . | cow |
| cow | animal that gives milk . | cow |
| cow | farm animal with horns . | giraffe |
| sheep | animal people raise for wool . | sheep |
| bus | large ride for many people . | car |
