# Think-then-Detect Commonsense Results

- n_samples: 98
- shard: 0/8

## Overall

| condition | hit@0.5_top1 | hit@0.5_topk | mean_iou_top1 |
|---|---:|---:|---:|
| baseline_direct | 0.3980 | 0.6531 | 0.3746 |
| baseline_translated | 0.6837 | 0.7959 | 0.6104 |
| thinkdet_direct | 0.4082 | 0.6633 | 0.3860 |
| thinkdet_translated | 0.6939 | 0.7857 | 0.6180 |

## Sample Translations

| object | original prompt | translated |
|---|---|---|
| person | someone you talk with . | cell phone |
| person | someone you ask questions . | cell phone |
| chair | seat for one person . | chair |
| giraffe | spotted animal seen on trips . | giraffe |
| zebra | wild animal behind a fence . | zebra |
| elephant | huge animal people see at zoos . | elephant |
| elephant | wild animal behind a fence . | elephant |
| bowl | round dish for food . | plate |
| couch | something used while watching shows . | remote control |
| pizza | baked food with toppings . | pizza |
| horse | animal people ride . | dog |
| horse | animal used for riding lessons . | horse |
| sandwich | quick meal between bread . | hot dog bun |
| car | something parked outside . | car |
| dining table | flat surface for dinner . | tray |
| motorcycle | vehicle people ride . | bicycle |
| motorcycle | something worn with a helmet . | helmet |
| cow | animal that gives milk . | giraffe |
| sheep | farm animal kept in groups . | sheep |
| bus | ride that stops often . | bus |
