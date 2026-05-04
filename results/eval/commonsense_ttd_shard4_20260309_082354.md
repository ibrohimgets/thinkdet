# Think-then-Detect Commonsense Results

- n_samples: 97
- shard: 4/8

## Overall

| condition | hit@0.5_top1 | hit@0.5_topk | mean_iou_top1 |
|---|---:|---:|---:|
| baseline_direct | 0.3918 | 0.6082 | 0.3785 |
| baseline_translated | 0.6289 | 0.7526 | 0.5762 |
| thinkdet_direct | 0.3918 | 0.6186 | 0.3768 |
| thinkdet_translated | 0.6392 | 0.7423 | 0.5871 |

## Sample Translations

| object | original prompt | translated |
|---|---|---|
| person | someone you walk beside . | cell phone |
| chair | furniture at a meal . | chair |
| giraffe | tall animal people see at zoos . | giraffe |
| giraffe | animal often behind a fence . | giraffe |
| zebra | animal with black and white stripes . | zebra |
| elephant | animal kids like watching . | elephant |
| bowl | container for cereal . | bowl |
| couch | something you sit on . | couch |
| couch | furniture for a living room . | chair |
| pizza | something shared at dinner . | pizza |
| horse | large animal with a mane . | dog |
| sandwich | something packed for lunch . | sandwich |
| car | something people drive . | car |
| car | way to travel to work . | bus |
| dining table | place where people eat together . | table |
| motorcycle | fast ride for one person . | bicycle |
| cow | large animal seen in fields . | cow |
| sheep | farm animal with wool . | sheep |
| sheep | fluffy animal behind a fence . | sheep |
| bus | vehicle used for commuting . | car |
