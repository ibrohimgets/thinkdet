# Think-then-Detect Commonsense Results

- n_samples: 98
- shard: 1/8

## Overall

| condition | hit@0.5_top1 | hit@0.5_topk | mean_iou_top1 |
|---|---:|---:|---:|
| baseline_direct | 0.3571 | 0.6327 | 0.3556 |
| baseline_translated | 0.6429 | 0.7551 | 0.5774 |
| thinkdet_direct | 0.3673 | 0.6327 | 0.3670 |
| thinkdet_translated | 0.6429 | 0.7551 | 0.5732 |

## Sample Translations

| object | original prompt | translated |
|---|---|---|
| person | someone who can help . | cell phone |
| person | someone you walk beside . | cell phone |
| chair | furniture at a meal . | chair |
| giraffe | tall animal people see at zoos . | giraffe |
| zebra | animal kids point at . | animal |
| elephant | animal with a long trunk . | elephant |
| elephant | animal kids like watching . | elephant |
| bowl | container for cereal . | bottle |
| couch | something you sit on . | chair |
| pizza | food for a quick meal . | pizza |
| horse | farm animal people feed . | dog |
| horse | large animal with a mane . | horse |
| sandwich | something packed for lunch . | hot dog |
| car | something people drive . | car |
| dining table | furniture for plates and cups . | tray |
| motorcycle | two-wheeled way to travel . | bicycle |
| motorcycle | fast ride for one person . | motorcycle |
| cow | large animal seen in fields . | giraffe |
| sheep | farm animal with wool . | sheep |
| bus | way to get to school . | bus |
