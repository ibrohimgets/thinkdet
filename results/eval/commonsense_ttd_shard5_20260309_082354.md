# Think-then-Detect Commonsense Results

- n_samples: 97
- shard: 5/8

## Overall

| condition | hit@0.5_top1 | hit@0.5_topk | mean_iou_top1 |
|---|---:|---:|---:|
| baseline_direct | 0.4124 | 0.6598 | 0.3886 |
| baseline_translated | 0.5773 | 0.7113 | 0.5242 |
| thinkdet_direct | 0.4227 | 0.6186 | 0.3977 |
| thinkdet_translated | 0.5979 | 0.7113 | 0.5418 |

## Sample Translations

| object | original prompt | translated |
|---|---|---|
| person | someone you talk with . | cell phone |
| chair | place to rest indoors . | chair |
| giraffe | animal with a very long neck . | giraffe |
| giraffe | spotted animal seen on trips . | giraffe |
| zebra | wild animal behind a fence . | giraffe |
| elephant | huge animal people see at zoos . | elephant |
| bowl | dish for snacks . | bowl |
| couch | soft furniture for relaxing . | couch |
| couch | something used while watching shows . | wii controller |
| pizza | baked food with toppings . | pizza |
| horse | animal people ride . | horse |
| sandwich | hand-held food for a meal . | sandwich |
| car | vehicle for getting around . | car |
| car | something parked outside . | bus |
| dining table | flat surface for dinner . | table |
| motorcycle | vehicle people ride . | motorcycle |
| cow | animal people feed on farms . | cow |
| sheep | animal seen in fields . | sheep |
| sheep | farm animal kept in groups . | sheep |
| bus | ride that stops often . | bicycle |
