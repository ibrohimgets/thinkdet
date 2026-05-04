# Think-then-Detect Commonsense Results

- n_samples: 97
- shard: 7/8

## Overall

| condition | hit@0.5_top1 | hit@0.5_topk | mean_iou_top1 |
|---|---:|---:|---:|
| baseline_direct | 0.4639 | 0.6804 | 0.4335 |
| baseline_translated | 0.7320 | 0.8351 | 0.6524 |
| thinkdet_direct | 0.4433 | 0.6907 | 0.4166 |
| thinkdet_translated | 0.7320 | 0.8351 | 0.6535 |

## Sample Translations

| object | original prompt | translated |
|---|---|---|
| person | someone you meet outside . | cell phone |
| chair | something you sit on . | chair |
| giraffe | animal often behind a fence . | giraffe |
| zebra | animal with black and white stripes . | zebra |
| zebra | animal often standing in groups . | giraffe |
| elephant | gray animal with big ears . | elephant |
| bowl | something used for soup . | bottle |
| couch | furniture for a living room . | furniture for a living room  |
| pizza | something shared at dinner . | pizza |
| pizza | sliceable food for parties . | pizza |
| horse | animal seen in fields . | horse |
| sandwich | food people eat for lunch . | hot dog |
| car | way to travel to work . | car |
| dining table | place where people eat together . | tray |
| dining table | spot for family meals . | dining table |
| motorcycle | ride used on roads . | motorcycle |
| cow | farm animal people raise . | giraffe |
| sheep | fluffy animal behind a fence . | sheep |
| bus | vehicle used for commuting . | bus |
| bus | public ride on city streets . | car |
